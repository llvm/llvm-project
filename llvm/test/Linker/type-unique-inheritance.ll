; RUN: llvm-link %S/Inputs/type-unique-inheritance-a.ll %S/Inputs/type-unique-inheritance-b.ll -S -o - | FileCheck %S/Inputs/type-unique-inheritance-a.ll
