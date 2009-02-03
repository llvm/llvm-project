//===-- Use.cpp - Implement the Use class ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the algorithm for finding the User of a Use.
//
//===----------------------------------------------------------------------===//

#include "llvm/User.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                         Use swap Implementation
//===----------------------------------------------------------------------===//

void Use::swap(Use &RHS) {
  Value *V1(Val);
  Value *V2(RHS.Val);
  if (V1 != V2) {
    if (V1) {
      removeFromList();
    }

    if (V2) {
      RHS.removeFromList();
      Val = V2;
      V2->addUse(*this);
    } else {
      Val = 0;
    }

    if (V1) {
      RHS.Val = V1;
      V1->addUse(RHS);
    } else {
      RHS.Val = 0;
    }
  }
}

//===----------------------------------------------------------------------===//
//                         Use getImpliedUser Implementation
//===----------------------------------------------------------------------===//

const Use *Use::getImpliedUser() const {
  const Use *Current = this;

  while (true) {
    unsigned Tag = (Current++)->Prev.getInt();
    switch (Tag) {
      case zeroDigitTag:
      case oneDigitTag:
        continue;

      case stopTag: {
        ++Current;
        ptrdiff_t Offset = 1;
        while (true) {
          unsigned Tag = Current->Prev.getInt();
          switch (Tag) {
            case zeroDigitTag:
            case oneDigitTag:
              ++Current;
              Offset = (Offset << 1) + Tag;
              continue;
            default:
              return Current + Offset;
          }
        }
      }

      case fullStopTag:
        return Current;
    }
  }
}

//===----------------------------------------------------------------------===//
//                         Use initTags Implementation
//===----------------------------------------------------------------------===//

Use *Use::initTags(Use * const Start, Use *Stop, ptrdiff_t Done) {
  ptrdiff_t Count = Done;
  while (Start != Stop) {
    --Stop;
    Stop->Val = 0;
    if (!Count) {
      Stop->Prev.setFromOpaqueValue(reinterpret_cast<Use**>(Done == 0
                                                            ? fullStopTag
                                                            : stopTag));
      ++Done;
      Count = Done;
    } else {
      Stop->Prev.setFromOpaqueValue(reinterpret_cast<Use**>(Count & 1));
      Count >>= 1;
      ++Done;
    }
  }

  return Start;
}

//===----------------------------------------------------------------------===//
//                         Use zap Implementation
//===----------------------------------------------------------------------===//

void Use::zap(Use *Start, const Use *Stop, bool del) {
  if (del) {
    while (Start != Stop) {
      (--Stop)->~Use();
    }
    ::operator delete(Start);
    return;
  }

  while (Start != Stop) {
    (Start++)->set(0);
  }
}

//===----------------------------------------------------------------------===//
//                         AugmentedUse layout struct
//===----------------------------------------------------------------------===//

struct AugmentedUse : Use {
  PointerIntPair<User*, 1, Tag> ref;
  AugmentedUse(); // not implemented
};


//===----------------------------------------------------------------------===//
//                         Use getUser Implementation
//===----------------------------------------------------------------------===//

User *Use::getUser() const {
  const Use *End = getImpliedUser();
  const PointerIntPair<User*, 1, Tag>& ref(
                                static_cast<const AugmentedUse*>(End - 1)->ref);
  User *She = ref.getPointer();
  return ref.getInt()
    ? She
    : (User*)End;
}

//===----------------------------------------------------------------------===//
//                         User allocHungoffUses Implementation
//===----------------------------------------------------------------------===//

Use *User::allocHungoffUses(unsigned N) const {
  Use *Begin = static_cast<Use*>(::operator new(sizeof(Use) * N
                                                + sizeof(AugmentedUse)
                                                - sizeof(Use)));
  Use *End = Begin + N;
  PointerIntPair<User*, 1, Tag>& ref(static_cast<AugmentedUse&>(End[-1]).ref);
  ref.setPointer(const_cast<User*>(this));
  ref.setInt(tagOne);
  return Use::initTags(Begin, End);
}

} // End llvm namespace
