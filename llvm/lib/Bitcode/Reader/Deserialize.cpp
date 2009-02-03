//==- Deserialize.cpp - Generic Object Serialization to Bitcode --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal methods used for object serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/Deserialize.h"

#ifdef DEBUG_BACKPATCH
#include "llvm/Support/Streams.h"
#endif

using namespace llvm;

Deserializer::Deserializer(BitstreamReader& stream)
  : Stream(stream), RecIdx(0), FreeList(NULL), AbbrevNo(0), RecordCode(0) {

    StreamStart = Stream.GetCurrentBitNo();
}

Deserializer::~Deserializer() {
  assert (RecIdx >= Record.size() && 
          "Still scanning bitcode record when deserialization completed.");
 
#ifdef DEBUG_BACKPATCH
  for (MapTy::iterator I=BPatchMap.begin(), E=BPatchMap.end(); I!=E; ++I)
    assert (I->first.hasFinalPtr() &&
            "Some pointers were not backpatched.");
#endif
}


bool Deserializer::inRecord() {
  if (Record.size() > 0) {
    if (RecIdx >= Record.size()) {
      RecIdx = 0;
      Record.clear();
      AbbrevNo = 0;
      return false;
    }
    else
      return true;
  }

  return false;
}

bool Deserializer::AdvanceStream() {
  assert (!inRecord() && 
          "Cannot advance stream.  Still processing a record.");
  
  if (AbbrevNo == bitc::ENTER_SUBBLOCK ||
      AbbrevNo >= bitc::UNABBREV_RECORD)
    return true;
  
  while (!Stream.AtEndOfStream()) {
    
    uint64_t Pos = Stream.GetCurrentBitNo();
    AbbrevNo = Stream.ReadCode();    
  
    switch (AbbrevNo) {        
      case bitc::ENTER_SUBBLOCK: {
        unsigned id = Stream.ReadSubBlockID();
        
        // Determine the extent of the block.  This is useful for jumping around
        // the stream.  This is hack: we read the header of the block, save
        // the length, and then revert the bitstream to a location just before
        // the block is entered.
        uint64_t BPos = Stream.GetCurrentBitNo();
        Stream.ReadVBR(bitc::CodeLenWidth); // Skip the code size.
        Stream.SkipToWord();
        unsigned NumWords = Stream.Read(bitc::BlockSizeWidth);
        Stream.JumpToBit(BPos);
                
        BlockStack.push_back(Location(Pos,id,NumWords));
        break;
      } 
        
      case bitc::END_BLOCK: {
        bool x = Stream.ReadBlockEnd();
        assert(!x && "Error at block end."); x=x;
        BlockStack.pop_back();
        continue;
      }
        
      case bitc::DEFINE_ABBREV:
        Stream.ReadAbbrevRecord();
        continue;

      default:
        break;
    }
    
    return true;
  }
  
  return false;
}

void Deserializer::ReadRecord() {

  while (AdvanceStream() && AbbrevNo == bitc::ENTER_SUBBLOCK) {
    assert (!BlockStack.empty());
    Stream.EnterSubBlock(BlockStack.back().BlockID);
    AbbrevNo = 0;
  }

  if (Stream.AtEndOfStream())
    return;
  
  assert (Record.empty());
  assert (AbbrevNo >= bitc::UNABBREV_RECORD);
  RecordCode = Stream.ReadRecord(AbbrevNo,Record);
  assert (Record.size() > 0);
}

void Deserializer::SkipBlock() {
  assert (!inRecord());

  if (AtEnd())
    return;

  AdvanceStream();  

  assert (AbbrevNo == bitc::ENTER_SUBBLOCK);
  BlockStack.pop_back();
  Stream.SkipBlock();

  AbbrevNo = 0;
  AdvanceStream();
}

bool Deserializer::SkipToBlock(unsigned BlockID) {
  assert (!inRecord());
  
  AdvanceStream();
  assert (AbbrevNo == bitc::ENTER_SUBBLOCK);
  
  unsigned BlockLevel = BlockStack.size();

  while (!AtEnd() &&
         BlockLevel == BlockStack.size() && 
         getCurrentBlockID() != BlockID)
    SkipBlock();

  return !(AtEnd() || BlockLevel != BlockStack.size());
}

Deserializer::Location Deserializer::getCurrentBlockLocation() {
  if (!inRecord())
    AdvanceStream();
  
  return BlockStack.back();
}

bool Deserializer::JumpTo(const Location& Loc) {
    
  assert (!inRecord());

  AdvanceStream();
  
  assert (!BlockStack.empty() || AtEnd());
    
  uint64_t LastBPos = StreamStart;
  
  while (!BlockStack.empty()) {
    
    LastBPos = BlockStack.back().BitNo;
    
    // Determine of the current block contains the location of the block
    // we are looking for.
    if (BlockStack.back().contains(Loc)) {
      // We found the enclosing block.  We must first POP it off to
      // destroy any accumulated context within the block scope.  We then
      // jump to the position of the block and enter it.
      Stream.JumpToBit(LastBPos);
      
      if (BlockStack.size() == Stream.BlockScope.size())
        Stream.PopBlockScope();

      BlockStack.pop_back();
      
      AbbrevNo = 0;
      AdvanceStream();      
      assert (AbbrevNo == bitc::ENTER_SUBBLOCK);
      
      Stream.EnterSubBlock(BlockStack.back().BlockID);
      break;
    }

    // This block does not contain the block we are looking for.  Pop it.
    if (BlockStack.size() == Stream.BlockScope.size())
      Stream.PopBlockScope();
    
    BlockStack.pop_back();

  }

  // Check if we have popped our way to the outermost scope.  If so,
  // we need to adjust our position.
  if (BlockStack.empty()) {
    assert (Stream.BlockScope.empty());
    
    Stream.JumpToBit(Loc.BitNo < LastBPos ? StreamStart : LastBPos);
    AbbrevNo = 0;
    AdvanceStream();
  }

  assert (AbbrevNo == bitc::ENTER_SUBBLOCK);
  assert (!BlockStack.empty());
  
  while (!AtEnd() && BlockStack.back() != Loc) {
    if (BlockStack.back().contains(Loc)) {
      Stream.EnterSubBlock(BlockStack.back().BlockID);
      AbbrevNo = 0;
      AdvanceStream();
      continue;
    }
    else
      SkipBlock();
  }
  
  if (AtEnd())
    return false;
  
  assert (BlockStack.back() == Loc);

  return true;
}

void Deserializer::Rewind() {
  while (!Stream.BlockScope.empty())
    Stream.PopBlockScope();
  
  while (!BlockStack.empty())
    BlockStack.pop_back();
  
  Stream.JumpToBit(StreamStart);
  AbbrevNo = 0;
}
  

unsigned Deserializer::getCurrentBlockID() { 
  if (!inRecord())
    AdvanceStream();
  
  return BlockStack.back().BlockID;
}

unsigned Deserializer::getRecordCode() {
  if (!inRecord()) {
    AdvanceStream();
    assert (AbbrevNo >= bitc::UNABBREV_RECORD);
    ReadRecord();
  }
  
  return RecordCode;
}

bool Deserializer::FinishedBlock(Location BlockLoc) {
  if (!inRecord())
    AdvanceStream();

  for (llvm::SmallVector<Location,8>::reverse_iterator
        I=BlockStack.rbegin(), E=BlockStack.rend(); I!=E; ++I)
      if (*I == BlockLoc)
        return false;
  
  return true;
}

unsigned Deserializer::getAbbrevNo() {
  if (!inRecord())
    AdvanceStream();
  
  return AbbrevNo;
}

bool Deserializer::AtEnd() {
  if (inRecord())
    return false;
  
  if (!AdvanceStream())
    return true;
  
  return false;
}

uint64_t Deserializer::ReadInt() {
  // FIXME: Any error recovery/handling with incomplete or bad files?
  if (!inRecord())
    ReadRecord();

  return Record[RecIdx++];
}

int64_t Deserializer::ReadSInt() {
  uint64_t x = ReadInt();
  int64_t magnitude = x >> 1;
  return x & 0x1 ? -magnitude : magnitude;
}

char* Deserializer::ReadCStr(char* cstr, unsigned MaxLen, bool isNullTerm) {
  if (cstr == NULL)
    MaxLen = 0; // Zero this just in case someone does something funny.
  
  unsigned len = ReadInt();

  assert (MaxLen == 0 || (len + (isNullTerm ? 1 : 0)) <= MaxLen);

  if (!cstr)
    cstr = new char[len + (isNullTerm ? 1 : 0)];
  
  assert (cstr != NULL);
  
  for (unsigned i = 0; i < len; ++i)
    cstr[i] = (char) ReadInt();
  
  if (isNullTerm)
    cstr[len] = '\0';
  
  return cstr;
}

void Deserializer::ReadCStr(std::vector<char>& buff, bool isNullTerm,
                            unsigned Idx) {
  
  unsigned len = ReadInt();

  // If Idx is beyond the current before size, reduce Idx to refer to the
  // element after the last element.
  if (Idx > buff.size())
    Idx = buff.size();

  buff.reserve(len+Idx);
  buff.resize(Idx);      
  
  for (unsigned i = 0; i < len; ++i)
    buff.push_back((char) ReadInt());
  
  if (isNullTerm)
    buff.push_back('\0');
}

void Deserializer::RegisterPtr(const SerializedPtrID& PtrId,
                               const void* Ptr) {
  
  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  assert (!HasFinalPtr(E) && "Pointer already registered.");

#ifdef DEBUG_BACKPATCH
  llvm::cerr << "RegisterPtr: " << PtrId << " => " << Ptr << "\n";
#endif 
  
  SetPtr(E,Ptr);
}

void Deserializer::ReadUIntPtr(uintptr_t& PtrRef, 
                               const SerializedPtrID& PtrId,
                               bool AllowBackpatch) {  
  if (PtrId == 0) {
    PtrRef = 0;
    return;
  }
  
  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  if (HasFinalPtr(E)) {
    PtrRef = GetFinalPtr(E);

#ifdef DEBUG_BACKPATCH
    llvm::cerr << "ReadUintPtr: " << PtrId
               << " <-- " <<  (void*) GetFinalPtr(E) << '\n';
#endif    
  }
  else {
    assert (AllowBackpatch &&
            "Client forbids backpatching for this pointer.");
    
#ifdef DEBUG_BACKPATCH
    llvm::cerr << "ReadUintPtr: " << PtrId << " (NO PTR YET)\n";
#endif
    
    // Register backpatch.  Check the freelist for a BPNode.
    BPNode* N;

    if (FreeList) {
      N = FreeList;
      FreeList = FreeList->Next;
    }
    else // No available BPNode.  Allocate one.
      N = (BPNode*) Allocator.Allocate<BPNode>();
    
    new (N) BPNode(GetBPNode(E),PtrRef);
    SetBPNode(E,N);
  }
}

uintptr_t Deserializer::ReadInternalRefPtr() {
  SerializedPtrID PtrId = ReadPtrID();
  
  assert (PtrId != 0 && "References cannot refer the NULL address.");

  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  assert (HasFinalPtr(E) &&
          "Cannot backpatch references.  Object must be already deserialized.");
  
  return GetFinalPtr(E);
}

void Deserializer::BPEntry::SetPtr(BPNode*& FreeList, void* P) {
  BPNode* Last = NULL;
  
  for (BPNode* N = Head; N != NULL; N=N->Next) {
    Last = N;
    N->PtrRef |= reinterpret_cast<uintptr_t>(P);
  }
  
  if (Last) {
    Last->Next = FreeList;
    FreeList = Head;
  }
  
  Ptr = const_cast<void*>(P);
}


#define INT_READ(TYPE)\
void SerializeTrait<TYPE>::Read(Deserializer& D, TYPE& X) {\
  X = (TYPE) D.ReadInt(); }

INT_READ(bool)
INT_READ(unsigned char)
INT_READ(unsigned short)
INT_READ(unsigned int)
INT_READ(unsigned long)

#define SINT_READ(TYPE)\
void SerializeTrait<TYPE>::Read(Deserializer& D, TYPE& X) {\
  X = (TYPE) D.ReadSInt(); }

INT_READ(signed char)
INT_READ(signed short)
INT_READ(signed int)
INT_READ(signed long)
