; This file is used with func-attrs-a.ll
; RUN: true

%struct.S0 = type <{ i8, i8, i8, i8 }>

define void @check0(ptr sret(%struct.S0) %agg.result, ptr byval(%struct.S0) %arg0, ptr %arg1, ptr byval(%struct.S0) %arg2) {
  ret void
}
