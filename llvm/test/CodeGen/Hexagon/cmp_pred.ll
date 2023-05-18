; RUN: llc -march=hexagon -hexagon-gen-mux-threshold=4 < %s | FileCheck %s
; Generate various cmpb instruction followed by if (p0) .. if (!p0)...
target triple = "hexagon"

define i32 @Func_3Ugt(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ugt i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3Uge(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp uge i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3Ult(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ult i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3Ule(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ule i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3Ueq(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp eq i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3Une(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ne i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3UneC(i32 %Enum_Par_Val) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ne i32 %Enum_Par_Val, 122
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3gt(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp sgt i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3ge(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp sge i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3lt(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp slt i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3le(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp sle i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3eq(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp eq i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3ne(i32 %Enum_Par_Val, i32 %pv2) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ne i32 %Enum_Par_Val, %pv2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3neC(i32 %Enum_Par_Val) nounwind readnone {
entry:
; CHECK-NOT: mux
  %cmp = icmp ne i32 %Enum_Par_Val, 122
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}
