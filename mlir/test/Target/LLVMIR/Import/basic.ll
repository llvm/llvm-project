; RUN: mlir-translate -import-llvm %s | FileCheck %s
; RUN: mlir-translate -import-llvm -mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-DBG

; CHECK-DBG: #[[UNKNOWN_LOC:.+]] = loc(unknown)

@global = external global double, align 8

; CHECK: llvm.func @fe(i32) -> f32
declare float @fe(i32)

; CHECK-LABEL: llvm.func internal @f1(%arg0: i64) -> i32 attributes {dso_local, passthrough = ["norecurse"]} {
; CHECK-DBG: llvm.func internal @f1(%arg0: i64 loc(unknown)) -> i32 attributes {dso_local, passthrough = ["norecurse"]} {
; CHECK: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK: %[[c1:[0-9]+]] = llvm.mlir.constant(true) : i1
; CHECK: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
; CHECK: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  %aa = inttoptr i64 %a to ptr
; CHECK-DBG: llvm.mlir.addressof @global : !llvm.ptr loc(#[[UNKNOWN_LOC]])
; %[[addrof:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
; %[[addrof2:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
; %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr
; %{{[0-9]+}} = llvm.ptrtoint %[[addrof2]] : !llvm.ptr to i64
; %{{[0-9]+}} = llvm.getelementptr %[[addrof]][%3] : (!llvm.ptr, i32) -> !llvm.ptr
  %bb = ptrtoint ptr @global to i64
  %cc = getelementptr double, ptr @global, i32 3
; CHECK: %[[b:[0-9]+]] = llvm.trunc %arg0 : i64 to i32
; CHECK-DBG: llvm.trunc %arg0 : i64 to i32 loc(#[[UNKNOWN_LOC]])
  %b = trunc i64 %a to i32
; CHECK: %[[c:[0-9]+]] = llvm.call @fe(%[[b]]) : (i32) -> f32
  %c = call float @fe(i32 %b)
; CHECK: %[[d:[0-9]+]] = llvm.fptosi %[[c]] : f32 to i32
  %d = fptosi float %c to i32
; CHECK: %[[e:[0-9]+]] = llvm.icmp "ne" %[[d]], %[[c2]] : i32
  %e = icmp ne i32 %d, 2
; CHECK: llvm.cond_br %[[e]], ^bb1, ^bb2
  br i1 %e, label %if.then, label %if.end

; CHECK: ^bb1:
if.then:
; CHECK: llvm.return %[[c42]] : i32
  ret i32 42

; CHECK: ^bb2:
if.end:
; CHECK: %[[orcond:[0-9]+]] = llvm.or %[[e]], %[[c1]] : i1
  %or.cond = or i1 %e, 1
; CHECK: llvm.return %[[c43]]
  ret i32 43
}
; CHECK-DBG: } loc(#[[UNKNOWN_LOC]])

; CHECK-LABEL: @hasGCFunction
; CHECK-SAME: garbageCollector = "statepoint-example"
define void @hasGCFunction() gc "statepoint-example" {
    ret void
}

; CHECK-LABEL: @useFreezeOp
define i32 @useFreezeOp(i32 %x) {
  ;CHECK: %{{[0-9]+}} = llvm.freeze %{{[0-9a-z]+}} : i32
  %1 = freeze i32 %x
  %2 = add i8 10, 10
  ;CHECK: %{{[0-9]+}} = llvm.freeze %{{[0-9]+}} : i8
  %3 = freeze i8 %2
  %poison = add nsw i1 0, undef
  ret i32 0
}

; Varadic function definition
%struct.va_list = type { ptr }

declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_copy.p0(ptr, ptr)
declare void @llvm.va_end.p0(ptr)

; CHECK-LABEL: llvm.func @variadic_function
define void @variadic_function(i32 %X, ...) {
  ; CHECK: %[[ALLOCA0:.+]] = llvm.alloca %{{.*}} x !llvm.struct<"struct.va_list", (ptr)> {{.*}} : (i32) -> !llvm.ptr
  %ap = alloca %struct.va_list
  ; CHECK: llvm.intr.vastart %[[ALLOCA0]]
  call void @llvm.va_start.p0(ptr %ap)

  ; CHECK: %[[ALLOCA1:.+]] = llvm.alloca %{{.*}} x !llvm.ptr {{.*}} : (i32) -> !llvm.ptr
  %aq = alloca ptr
  ; CHECK: llvm.intr.vacopy %[[ALLOCA0]] to %[[ALLOCA1]]
  call void @llvm.va_copy.p0(ptr %aq, ptr %ap)
  ; CHECK: llvm.intr.vaend %[[ALLOCA1]]
  call void @llvm.va_end.p0(ptr %aq)

  ; CHECK: llvm.intr.vaend %[[ALLOCA0]]
  call void @llvm.va_end.p0(ptr %ap)
  ; CHECK: llvm.return
  ret void
}
