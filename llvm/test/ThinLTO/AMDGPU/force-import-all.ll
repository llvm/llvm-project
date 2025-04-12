; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary %s -o %t.main.bc
; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary %p/Inputs/in-0.ll -o %t.in.0.bc
; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary %p/Inputs/in-1.ll -o %t.in.1.bc
; RUN: llvm-lto -thinlto-action=run -force-import-all %t.main.bc %t.in.0.bc %t.in.1.bc --thinlto-save-temps=%t.2.
; RUN: llvm-dis %t.2.0.3.imported.bc -o - | FileCheck --check-prefix=MOD1 %s
; RUN: llvm-dis %t.2.1.3.imported.bc -o - | FileCheck --check-prefix=MOD2 %s
; RUN: llvm-dis %t.2.2.3.imported.bc -o - | FileCheck --check-prefix=MOD3 %s

define void @f0(ptr %p) #0 {
entry:
  call void @f1(ptr %p)
  call void @f2(ptr %p)
  ret void
}

define weak hidden void @weak_common(ptr %v) #0 {
entry:
  store i32 12345, ptr %v
  ret void
}

declare void @f1(ptr)

declare void @f2(ptr)

attributes #0 = { noinline }

; MOD1: define available_externally void @f1
; MOD1: define available_externally void @f2

; MOD2: define void @f1
; MOD2: define available_externally hidden void @weak_common

; MOD3: define void @f2
; MOD3: define available_externally hidden void @weak_common
