//===-- CodeGen/MachineInstr.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the machine function pass registry for register allocators
// and instruction schedulers.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassRegistry.h"

using namespace llvm;


/// Add - Adds a function pass to the registration list.
///
void MachinePassRegistry::Add(MachinePassRegistryNode *Node) {
  Node->setNext(List);
  List = Node;
  if (Listener) Listener->NotifyAdd(Node->getName(),
                                    Node->getCtor(),
                                    Node->getDescription());
}


/// Remove - Removes a function pass from the registration list.
///
void MachinePassRegistry::Remove(MachinePassRegistryNode *Node) {
  for (MachinePassRegistryNode **I = &List; *I; I = (*I)->getNextAddress()) {
    if (*I == Node) {
      if (Listener) Listener->NotifyRemove(Node->getName());
      *I = (*I)->getNext();
      break;
    }
  }
}
