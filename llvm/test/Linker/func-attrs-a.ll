; RUN: llvm-link %s %p/func-attrs-b.ll -S -o - | FileCheck %s
; PR2382

; CHECK: call void @check0(%struct.S0* sret(%struct.S0) null, %struct.S0* byval(%struct.S0) align 4 null, %struct.S0* align 4 null, %struct.S0* byval(%struct.S0) align 4 null)
; CHECK: define void @check0(%struct.S0* sret(%struct.S0) %agg.result, %struct.S0* byval(%struct.S0) %arg0, %struct.S0* %arg1, %struct.S0* byval(%struct.S0) %arg2)

%struct.S0 = type <{ i8, i8, i8, i8 }>

define void @a() {
  call void @check0(%struct.S0* sret(%struct.S0) null, %struct.S0* byval(%struct.S0) align 4 null, %struct.S0* align 4 null, %struct.S0* byval(%struct.S0) align 4 null)
  ret void
}

declare void @check0(%struct.S0*, %struct.S0*, %struct.S0*, %struct.S0*)
