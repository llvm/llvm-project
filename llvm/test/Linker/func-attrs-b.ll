; This file is used with func-attrs-a.ll
; RUN: true

%struct.S0 = type <{ i8, i8, i8, i8 }>

define void @check0(%struct.S0* sret(%struct.S0) %agg.result, %struct.S0* byval(%struct.S0) %arg0, %struct.S0* %arg1, %struct.S0* byval(%struct.S0) %arg2) {
  ret void
}
