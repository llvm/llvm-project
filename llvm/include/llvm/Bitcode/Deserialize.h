//=- Deserialize.h - Generic Object Deserialization from Bitcode --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generic object deserialization from
// LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE_INPUT
#define LLVM_BITCODE_SERIALIZE_INPUT

#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/Serialization.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
  
class Deserializer {  

  //===----------------------------------------------------------===//
  // Internal type definitions.
  //===----------------------------------------------------------===//
  
  struct BPNode {
    BPNode* Next;
    uintptr_t& PtrRef;
    
    BPNode(BPNode* n, uintptr_t& pref) 
      : Next(n), PtrRef(pref) {
        PtrRef = 0;
      }
  };
  
  struct BPEntry { 
    union { BPNode* Head; void* Ptr; };
    
    BPEntry() : Head(NULL) {}
    
    static inline bool isPod() { return true; }
    
    void SetPtr(BPNode*& FreeList, void* P);    
  };  
  
  class BPKey {
    unsigned Raw;
    
  public:
    BPKey(SerializedPtrID PtrId) : Raw(PtrId << 1) { assert (PtrId > 0); }
    BPKey(unsigned code, unsigned) : Raw(code) {}
    
    void MarkFinal() { Raw |= 0x1; }
    bool hasFinalPtr() const { return Raw & 0x1 ? true : false; }
    SerializedPtrID getID() const { return Raw >> 1; }
    
    static inline BPKey getEmptyKey() { return BPKey(0,0); }
    static inline BPKey getTombstoneKey() { return BPKey(1,0); }
    static inline unsigned getHashValue(const BPKey& K) { return K.Raw & ~0x1; }

    static bool isEqual(const BPKey& K1, const BPKey& K2) {
      return (K1.Raw ^ K2.Raw) & ~0x1 ? false : true;
    }
    
    static bool isPod() { return true; }
  };
  
  typedef llvm::DenseMap<BPKey,BPEntry,BPKey,BPEntry> MapTy;

  //===----------------------------------------------------------===//
  // Publicly visible types.
  //===----------------------------------------------------------===//
  
public:  
  struct Location {
    uint64_t BitNo;
    unsigned BlockID;
    unsigned NumWords;
    
    Location(uint64_t bit, unsigned bid, unsigned words) 
    : BitNo(bit), BlockID(bid), NumWords(words) {}
    
    Location() : BitNo(0), BlockID(0), NumWords(0) {}

    Location& operator=(Location& RHS) {
      BitNo = RHS.BitNo;
      BlockID = RHS.BlockID;
      NumWords = RHS.NumWords;
      return *this;
    }
    
    bool operator==(const Location& RHS) const { return BitNo == RHS.BitNo; }    
    bool operator!=(const Location& RHS) const { return BitNo != RHS.BitNo; }
    
    bool contains(const Location& RHS) const {
      if (RHS.BitNo < BitNo)
        return false;

      if ((RHS.BitNo - BitNo) >> 5 < NumWords)
        return true;
      
      return false;
    }
  };
  
  //===----------------------------------------------------------===//
  // Internal data members.
  //===----------------------------------------------------------===//

private:
  BitstreamReader& Stream;
  SmallVector<uint64_t,20> Record;
  unsigned RecIdx;
  BumpPtrAllocator Allocator;
  BPNode* FreeList;
  MapTy BPatchMap;
  llvm::SmallVector<Location,8> BlockStack;
  unsigned AbbrevNo;
  unsigned RecordCode;
  uint64_t StreamStart;
  
  //===----------------------------------------------------------===//
  // Public Interface.
  //===----------------------------------------------------------===//
  
public:  
  Deserializer(BitstreamReader& stream);
  ~Deserializer();

  uint64_t ReadInt();
  int64_t ReadSInt();
  SerializedPtrID ReadPtrID() { return (SerializedPtrID) ReadInt(); }
  
  
  bool ReadBool() {
    return ReadInt() ? true : false;
  }

  template <typename T>
  inline T& Read(T& X) {
    SerializeTrait<T>::Read(*this,X);
    return X;
  }

  template <typename T>
  inline T* Create() {
    return SerializeTrait<T>::Create(*this);
  }
  
  char* ReadCStr(char* cstr = NULL, unsigned MaxLen=0, bool isNullTerm=true);
  void ReadCStr(std::vector<char>& buff, bool isNullTerm=false, unsigned Idx=0);

  template <typename T>
  inline T* ReadOwnedPtr(bool AutoRegister = true) {
    SerializedPtrID PtrID = ReadPtrID();    

    if (!PtrID)
      return NULL;
    
    T* x = SerializeTrait<T>::Create(*this);

    if (AutoRegister)
      RegisterPtr(PtrID,x);
    
    return x;
  }

  template <typename T, typename Arg1>
  inline T* ReadOwnedPtr(Arg1& arg1, bool AutoRegister = true) {
    SerializedPtrID PtrID = ReadPtrID();

    if (!PtrID)
      return NULL;

    T* x = SerializeTrait<T>::Create(*this, arg1);

    if (AutoRegister)
      RegisterPtr(PtrID,x);

    return x;
  }

  template <typename T>
  inline void ReadOwnedPtr(T*& Ptr, bool AutoRegister = true) {
    Ptr = ReadOwnedPtr<T>(AutoRegister);
  }
  
  template <typename T1, typename T2>
  void BatchReadOwnedPtrs(T1*& P1, T2*& P2,
                          bool A1=true, bool A2=true) {

    SerializedPtrID ID1 = ReadPtrID();
    SerializedPtrID ID2 = ReadPtrID();

    P1 = (ID1) ? SerializeTrait<T1>::Create(*this) : NULL;
    if (ID1 && A1) RegisterPtr(ID1,P1);

    P2 = (ID2) ? SerializeTrait<T2>::Create(*this) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);
  }

  template <typename T1, typename T2, typename Arg1>
  void BatchReadOwnedPtrs(T1*& P1, T2*& P2, Arg1& arg1,
                          bool A1=true, bool A2=true) {

    SerializedPtrID ID1 = ReadPtrID();
    SerializedPtrID ID2 = ReadPtrID();

    P1 = (ID1) ? SerializeTrait<T1>::Create(*this, arg1) : NULL;
    if (ID1 && A1) RegisterPtr(ID1,P1);

    P2 = (ID2) ? SerializeTrait<T2>::Create(*this, arg1) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);
  }

  template <typename T1, typename T2, typename T3>
  void BatchReadOwnedPtrs(T1*& P1, T2*& P2, T3*& P3,
                          bool A1=true, bool A2=true, bool A3=true) {
    
    SerializedPtrID ID1 = ReadPtrID();
    SerializedPtrID ID2 = ReadPtrID();
    SerializedPtrID ID3 = ReadPtrID();
    
    P1 = (ID1) ? SerializeTrait<T1>::Create(*this) : NULL;
    if (ID1 && A1) RegisterPtr(ID1,P1);    
    
    P2 = (ID2) ? SerializeTrait<T2>::Create(*this) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);
    
    P3 = (ID3) ? SerializeTrait<T3>::Create(*this) : NULL;
    if (ID3 && A3) RegisterPtr(ID3,P3);
  }

  template <typename T1, typename T2, typename T3, typename Arg1>
  void BatchReadOwnedPtrs(T1*& P1, T2*& P2, T3*& P3, Arg1& arg1,
                          bool A1=true, bool A2=true, bool A3=true) {

    SerializedPtrID ID1 = ReadPtrID();
    SerializedPtrID ID2 = ReadPtrID();
    SerializedPtrID ID3 = ReadPtrID();

    P1 = (ID1) ? SerializeTrait<T1>::Create(*this, arg1) : NULL;
    if (ID1 && A1) RegisterPtr(ID1,P1);

    P2 = (ID2) ? SerializeTrait<T2>::Create(*this, arg1) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);

    P3 = (ID3) ? SerializeTrait<T3>::Create(*this, arg1) : NULL;
    if (ID3 && A3) RegisterPtr(ID3,P3);
  }

  template <typename T>
  void BatchReadOwnedPtrs(unsigned NumPtrs, T** Ptrs, bool AutoRegister=true) {
    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;
    
    for (unsigned i = 0; i < NumPtrs; ++i)
      BatchIDVec.push_back(ReadPtrID());
    
    for (unsigned i = 0; i < NumPtrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];
      
      T* p = PtrID ? SerializeTrait<T>::Create(*this) : NULL;
      
      if (PtrID && AutoRegister)
        RegisterPtr(PtrID,p);
      
      Ptrs[i] = p;
    }
  }
  
  template <typename T, typename Arg1>
  void BatchReadOwnedPtrs(unsigned NumPtrs, T** Ptrs, Arg1& arg1,
                          bool AutoRegister=true) {

    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;

    for (unsigned i = 0; i < NumPtrs; ++i)
      BatchIDVec.push_back(ReadPtrID());

    for (unsigned i = 0; i < NumPtrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];

      T* p = PtrID ? SerializeTrait<T>::Create(*this, arg1) : NULL;

      if (PtrID && AutoRegister)
        RegisterPtr(PtrID,p);

      Ptrs[i] = p;
    }
  }

  template <typename T1, typename T2>
  void BatchReadOwnedPtrs(unsigned NumT1Ptrs, T1** Ptrs, T2*& P2,
                          bool A1=true, bool A2=true) {
    
    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;

    for (unsigned i = 0; i < NumT1Ptrs; ++i)
      BatchIDVec.push_back(ReadPtrID());
    
    SerializedPtrID ID2 = ReadPtrID();
    
    for (unsigned i = 0; i < NumT1Ptrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];
      
      T1* p = PtrID ? SerializeTrait<T1>::Create(*this) : NULL;
      
      if (PtrID && A1)
        RegisterPtr(PtrID,p);
      
      Ptrs[i] = p;
    }
    
    P2 = (ID2) ? SerializeTrait<T2>::Create(*this) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);    
  }    
  
  template <typename T1, typename T2, typename Arg1>
  void BatchReadOwnedPtrs(unsigned NumT1Ptrs, T1** Ptrs, T2*& P2, Arg1& arg1,
                          bool A1=true, bool A2=true) {

    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;

    for (unsigned i = 0; i < NumT1Ptrs; ++i)
      BatchIDVec.push_back(ReadPtrID());

    SerializedPtrID ID2 = ReadPtrID();

    for (unsigned i = 0; i < NumT1Ptrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];

      T1* p = PtrID ? SerializeTrait<T1>::Create(*this, arg1) : NULL;

      if (PtrID && A1)
        RegisterPtr(PtrID,p);

      Ptrs[i] = p;
    }

    P2 = (ID2) ? SerializeTrait<T2>::Create(*this, arg1) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);
  }

  template <typename T1, typename T2, typename T3>
  void BatchReadOwnedPtrs(unsigned NumT1Ptrs, T1** Ptrs, 
                          T2*& P2, T3*& P3,
                          bool A1=true, bool A2=true, bool A3=true) {
    
    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;
    
    for (unsigned i = 0; i < NumT1Ptrs; ++i)
      BatchIDVec.push_back(ReadPtrID());
    
    SerializedPtrID ID2 = ReadPtrID();
    SerializedPtrID ID3 = ReadPtrID();    
    
    for (unsigned i = 0; i < NumT1Ptrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];
      
      T1* p = PtrID ? SerializeTrait<T1>::Create(*this) : NULL;
      
      if (PtrID && A1)
        RegisterPtr(PtrID,p);
      
      Ptrs[i] = p;
    }
    
    P2 = (ID2) ? SerializeTrait<T2>::Create(*this) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);
    
    P3 = (ID3) ? SerializeTrait<T3>::Create(*this) : NULL;
    if (ID3 && A3) RegisterPtr(ID3,P3);    
  }    

  template <typename T1, typename T2, typename T3, typename Arg1>
  void BatchReadOwnedPtrs(unsigned NumT1Ptrs, T1** Ptrs,
                          T2*& P2, T3*& P3, Arg1& arg1,
                          bool A1=true, bool A2=true, bool A3=true) {

    llvm::SmallVector<SerializedPtrID,10> BatchIDVec;

    for (unsigned i = 0; i < NumT1Ptrs; ++i)
      BatchIDVec.push_back(ReadPtrID());

    SerializedPtrID ID2 = ReadPtrID();
    SerializedPtrID ID3 = ReadPtrID();

    for (unsigned i = 0; i < NumT1Ptrs; ++i) {
      SerializedPtrID& PtrID = BatchIDVec[i];

      T1* p = PtrID ? SerializeTrait<T1>::Create(*this, arg1) : NULL;

      if (PtrID && A1)
        RegisterPtr(PtrID,p);

      Ptrs[i] = p;
    }

    P2 = (ID2) ? SerializeTrait<T2>::Create(*this, arg1) : NULL;
    if (ID2 && A2) RegisterPtr(ID2,P2);

    P3 = (ID3) ? SerializeTrait<T3>::Create(*this, arg1) : NULL;
    if (ID3 && A3) RegisterPtr(ID3,P3);
  }

  template <typename T>
  void ReadPtr(T*& PtrRef, bool AllowBackpatch = true) {
    ReadUIntPtr(reinterpret_cast<uintptr_t&>(PtrRef), AllowBackpatch);
  }
  
  template <typename T>
  void ReadPtr(const T*& PtrRef, bool AllowBackpatch = true) {
    ReadPtr(const_cast<T*&>(PtrRef), AllowBackpatch);
  }
  
  
  template <typename T>
  void ReadPtr(T*& PtrRef, const SerializedPtrID& PtrID, 
               bool AllowBackpatch = true) {
    ReadUIntPtr(reinterpret_cast<uintptr_t&>(PtrRef), PtrID, AllowBackpatch);
  }
  
  template <typename T>
  void ReadPtr(const T*& PtrRef, const SerializedPtrID& PtrID, 
               bool AllowBackpatch = true) {
    
    ReadPtr(const_cast<T*&>(PtrRef), PtrID, AllowBackpatch);
  }
  
  template <typename T>
  T* ReadPtr() { T* x = 0; ReadPtr<T>(x,false); return x; }

  void ReadUIntPtr(uintptr_t& PtrRef, const SerializedPtrID& PtrID, 
                   bool AllowBackpatch = true);
  
  void ReadUIntPtr(uintptr_t& PtrRef, bool AllowBackpatch = true) {
    ReadUIntPtr(PtrRef,ReadPtrID(),AllowBackpatch);
  }
  
  template <typename T>
  T& ReadRef() {
    T* p = reinterpret_cast<T*>(ReadInternalRefPtr());
    return *p;
  }

  void RegisterPtr(const SerializedPtrID& PtrID, const void* Ptr);
  
  void RegisterPtr(const void* Ptr) {
    RegisterPtr(ReadPtrID(),Ptr);
  }
  
  template<typename T>
  void RegisterRef(const T& x) {
    RegisterPtr(&x);
  }
  
  template<typename T>
  void RegisterRef(const SerializedPtrID& PtrID, const T& x) {
    RegisterPtr(PtrID,&x);
  }  
  
  Location getCurrentBlockLocation();
  unsigned getCurrentBlockID();
  unsigned getAbbrevNo();
  
  bool FinishedBlock(Location BlockLoc);
  bool JumpTo(const Location& BlockLoc);
  void Rewind();
  
  bool AtEnd();
  bool inRecord();
  void SkipBlock();
  bool SkipToBlock(unsigned BlockID);
  
  unsigned getRecordCode();
  
  BitstreamReader& getStream() { return Stream; }
  
private:
  bool AdvanceStream();  
  void ReadRecord();
  
  uintptr_t ReadInternalRefPtr();
  
  static inline bool HasFinalPtr(MapTy::value_type& V) {
    return V.first.hasFinalPtr();
  }
  
  static inline uintptr_t GetFinalPtr(MapTy::value_type& V) {
    return reinterpret_cast<uintptr_t>(V.second.Ptr);
  }
  
  static inline BPNode* GetBPNode(MapTy::value_type& V) {
    return V.second.Head;
  }
    
  static inline void SetBPNode(MapTy::value_type& V, BPNode* N) {
    V.second.Head = N;
  }
  
  void SetPtr(MapTy::value_type& V, const void* P) {
    V.first.MarkFinal();
    V.second.SetPtr(FreeList,const_cast<void*>(P));
  }
};
    
} // end namespace llvm

#endif
