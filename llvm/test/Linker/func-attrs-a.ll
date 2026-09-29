; RUN: llvm-link %s %p/func-attrs-b.ll -S -o - | FileCheck %s
; PR2382

; CHECK: call void @check0(ptr sret(%struct.S0) null, ptr byval(%struct.S0) align 4 null, ptr align 4 null, ptr byval(%struct.S0) align 4 null)
; CHECK: define void @check0(ptr sret(%struct.S0) %agg.result, ptr byval(%struct.S0) %arg0, ptr %arg1, ptr byval(%struct.S0) %arg2)

%struct.S0 = type <{ i8, i8, i8, i8 }>

define void @a() {
  call void @check0(ptr sret(%struct.S0) null, ptr byval(%struct.S0) align 4 null, ptr align 4 null, ptr byval(%struct.S0) align 4 null)
  ret void
}

declare void @check0(ptr, ptr, ptr, ptr)
