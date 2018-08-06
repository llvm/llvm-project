; RUN: llc -march=hexagon -hexagon-gen-mux-threshold=4 < %s | FileCheck %s
; Generate various cmpb instruction followed by if (p0) .. if (!p0)...
target triple = "hexagon"

@Enum_global = external global i8

define i32 @Func_3(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 255
  %cmp = icmp eq i32 %conv, 2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3b(i32) nounwind readonly {
entry:
; CHECK-NOT: mux
  %1 = load i8, i8* @Enum_global, align 1
  %2 = trunc i32 %0 to i8
  %cmp = icmp ne i8 %1, %2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3c(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 255
  %cmp = icmp eq i32 %conv, 2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3d(i32) nounwind readonly {
entry:
; CHECK-NOT: mux
  %1 = load i8, i8* @Enum_global, align 1
  %2 = trunc i32 %0 to i8
  %cmp = icmp eq i8 %1, %2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3e(i32) nounwind readonly {
entry:
; CHECK-NOT: mux
  %1 = load i8, i8* @Enum_global, align 1
  %2 = trunc i32 %0 to i8
  %cmp = icmp eq i8 %1, %2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3f(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 255
  %cmp = icmp ugt i32 %conv, 2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3g(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 255
  %cmp = icmp ult i32 %conv, 3
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3h(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 254
  %cmp = icmp ult i32 %conv, 2
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}

define i32 @Func_3i(i32) nounwind readnone {
entry:
; CHECK-NOT: mux
  %conv = and i32 %0, 254
  %cmp = icmp ugt i32 %conv, 1
  %selv = zext i1 %cmp to i32
  ret i32 %selv
}
