; RUN: mlir-translate -opaque-pointers=0 -import-llvm %s | FileCheck %s
; RUN: mlir-translate -opaque-pointers=0 -import-llvm -mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-DBG

; CHECK-DBG: #[[MODULELOC:.+]] = loc({{.*}}basic.ll{{.*}}:0:0)

@global = external global double, align 8

; CHECK: llvm.func @fe(i32) -> f32
declare float @fe(i32)

; FIXME: function attributes.
; CHECK-LABEL: llvm.func internal @f1(%arg0: i64) -> i32 attributes {dso_local} {
; CHECK-DBG: llvm.func internal @f1(%arg0: i64 loc({{.*}}basic.ll{{.*}}:0:0)) -> i32 attributes {dso_local} {
; CHECK: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK: %[[c1:[0-9]+]] = llvm.mlir.constant(true) : i1
; CHECK: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
; CHECK: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
  %aa = inttoptr i64 %a to i64*
; CHECK-DBG: llvm.mlir.addressof @global : !llvm.ptr<f64> loc(#[[MODULELOC]])
; %[[addrof:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr<f64>
; %[[addrof2:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr<f64>
; %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
; %{{[0-9]+}} = llvm.ptrtoint %[[addrof2]] : !llvm.ptr<f64> to i64
; %{{[0-9]+}} = llvm.getelementptr %[[addrof]][%3] : (!llvm.ptr<f64>, i32) -> !llvm.ptr<f64>
  %bb = ptrtoint double* @global to i64
  %cc = getelementptr double, double* @global, i32 3
; CHECK: %[[b:[0-9]+]] = llvm.trunc %arg0 : i64 to i32
; CHECK-DBG: llvm.trunc %arg0 : i64 to i32 loc(#[[MODULELOC]])
  %b = trunc i64 %a to i32
; CHECK: %[[c:[0-9]+]] = llvm.call @fe(%[[b]]) : (i32) -> f32
  %c = call float @fe(i32 %b)
; CHECK: %[[d:[0-9]+]] = llvm.fptosi %[[c]] : f32 to i32
  %d = fptosi float %c to i32
; FIXME: icmp should return i1.
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
; CHECK-DBG: } loc(#[[MODULELOC]])


@_ZTIi = external dso_local constant i8*
@_ZTIii= external dso_local constant i8**
declare void @foo(i8*)
declare i8* @bar(i8*)
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @invokeLandingpad
define i32 @invokeLandingpad() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  ; CHECK: %[[a1:[0-9]+]] = llvm.bitcast %{{[0-9]+}} : !llvm.ptr<ptr<ptr<i8>>> to !llvm.ptr<i8>
  ; CHECK: %[[a3:[0-9]+]] = llvm.alloca %{{[0-9]+}} x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr<i8>
  %1 = alloca i8
  ; CHECK: llvm.invoke @foo(%[[a3]]) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> ()
  invoke void @foo(i8* %1) to label %4 unwind label %2

; CHECK: ^bb1:
  ; CHECK: %{{[0-9]+}} = llvm.landingpad (catch %{{[0-9]+}} : !llvm.ptr<ptr<i8>>) (catch %[[a1]] : !llvm.ptr<i8>) (filter %{{[0-9]+}} : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>, i32)>
  %3 = landingpad { i8*, i32 } catch i8** @_ZTIi catch i8* bitcast (i8*** @_ZTIii to i8*)
  ; FIXME: Change filter to a constant array once they are handled.
  ; Currently, even though it parses this, LLVM module is broken
          filter [1 x i8] [i8 1]
  resume { i8*, i32 } %3

; CHECK: ^bb2:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 1

; CHECK: ^bb3:
  ; CHECK: %{{[0-9]+}} = llvm.invoke @bar(%[[a3]]) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  %6 = invoke i8* @bar(i8* %1) to label %4 unwind label %2

; CHECK: ^bb4:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 0
}

; CHECK-LABEL: @hasGCFunction
; CHECK-SAME: garbageCollector = "statepoint-example"
define void @hasGCFunction() gc "statepoint-example" {
    ret void
}

;CHECK-LABEL: @useFreezeOp
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
%struct.va_list = type { i8* }

declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)

; CHECK-LABEL: llvm.func @variadic_function
define void @variadic_function(i32 %X, ...) {
  ; CHECK: %[[ALLOCA0:.+]] = llvm.alloca %{{.*}} x !llvm.struct<"struct.va_list", (ptr<i8>)> {{.*}} : (i32) -> !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>>
  %ap = alloca %struct.va_list
  ; CHECK: %[[CAST0:.+]] = llvm.bitcast %[[ALLOCA0]] : !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>> to !llvm.ptr<i8>
  %ap2 = bitcast %struct.va_list* %ap to i8*
  ; CHECK: llvm.intr.vastart %[[CAST0]]
  call void @llvm.va_start(i8* %ap2)

  ; CHECK: %[[ALLOCA1:.+]] = llvm.alloca %{{.*}} x !llvm.ptr<i8> {{.*}} : (i32) -> !llvm.ptr<ptr<i8>>
  %aq = alloca i8*
  ; CHECK: %[[CAST1:.+]] = llvm.bitcast %[[ALLOCA1]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %aq2 = bitcast i8** %aq to i8*
  ; CHECK: llvm.intr.vacopy %[[CAST0]] to %[[CAST1]]
  call void @llvm.va_copy(i8* %aq2, i8* %ap2)
  ; CHECK: llvm.intr.vaend %[[CAST1]]
  call void @llvm.va_end(i8* %aq2)

  ; CHECK: llvm.intr.vaend %[[CAST0]]
  call void @llvm.va_end(i8* %ap2)
  ; CHECK: llvm.return
  ret void
}
