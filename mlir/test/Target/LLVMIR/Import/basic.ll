; RUN: mlir-translate -import-llvm %s | FileCheck %s
; RUN: mlir-translate -import-llvm -mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-DBG

; CHECK-DBG: #[[MODULELOC:.+]] = loc({{.*}}basic.ll{{.*}}:0:0)

@global = external global double, align 8

; CHECK: llvm.func @fe(i32) -> f32
declare float @fe(i32)

; CHECK-LABEL: llvm.func internal @f1(%arg0: i64) -> i32 attributes {dso_local, passthrough = ["norecurse"]} {
; CHECK-DBG: llvm.func internal @f1(%arg0: i64 loc({{.*}}basic.ll{{.*}}:0:0)) -> i32 attributes {dso_local, passthrough = ["norecurse"]} {
; CHECK: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK: %[[c1:[0-9]+]] = llvm.mlir.constant(true) : i1
; CHECK: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
; CHECK: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  %aa = inttoptr i64 %a to ptr
; CHECK-DBG: llvm.mlir.addressof @global : !llvm.ptr loc(#[[MODULELOC]])
; %[[addrof:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
; %[[addrof2:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
; %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr
; %{{[0-9]+}} = llvm.ptrtoint %[[addrof2]] : !llvm.ptr to i64
; %{{[0-9]+}} = llvm.getelementptr %[[addrof]][%3] : (!llvm.ptr, i32) -> !llvm.ptr
  %bb = ptrtoint ptr @global to i64
  %cc = getelementptr double, ptr @global, i32 3
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


@_ZTIi = external dso_local constant ptr
@_ZTIii= external dso_local constant ptr
declare void @foo(ptr)
declare ptr @bar(ptr)
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @invokeLandingpad
define i32 @invokeLandingpad() personality ptr @__gxx_personality_v0 {
  ; CHECK: %[[a1:[0-9]+]] = llvm.mlir.addressof @_ZTIii : !llvm.ptr
  ; CHECK: %[[a3:[0-9]+]] = llvm.alloca %{{[0-9]+}} x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
  %1 = alloca i8
  ; CHECK: llvm.invoke @foo(%[[a3]]) to ^bb2 unwind ^bb1 : (!llvm.ptr) -> ()
  invoke void @foo(ptr %1) to label %4 unwind label %2

; CHECK: ^bb1:
  ; CHECK: %{{[0-9]+}} = llvm.landingpad (catch %{{[0-9]+}} : !llvm.ptr) (catch %[[a1]] : !llvm.ptr) (filter %{{[0-9]+}} : !llvm.array<1 x i1>) : !llvm.struct<(ptr, i32)>
  %3 = landingpad { ptr, i32 } catch ptr @_ZTIi catch ptr @_ZTIii
          filter [1 x i1] [i1 1]
  resume { ptr, i32 } %3

; CHECK: ^bb2:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 1

; CHECK: ^bb3:
  ; CHECK: %{{[0-9]+}} = llvm.invoke @bar(%[[a3]]) to ^bb2 unwind ^bb1 : (!llvm.ptr) -> !llvm.ptr
  %6 = invoke ptr @bar(ptr %1) to label %4 unwind label %2

; CHECK: ^bb4:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 0
}

declare i32 @foo2()

; CHECK-LABEL: @invokePhi
; CHECK-SAME:            (%[[cond:.*]]: i1) -> i32
define i32 @invokePhi(i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK: llvm.cond_br %[[cond]], ^[[bb1:.*]], ^[[bb2:.*]]
  br i1 %cond, label %call, label %nocall
; CHECK: ^[[bb1]]:
call:
  ; CHECK: %[[invoke:.*]] = llvm.invoke @foo2() to ^[[bb3:.*]] unwind ^[[bb5:.*]] : () -> i32
  %invoke = invoke i32 @foo2() to label %bb0 unwind label %bb1
; CHECK: ^[[bb2]]:
nocall:
  ; CHECK: llvm.br ^[[bb4:.*]](%[[c0]] : i32)
  br label %bb0
; CHECK: ^[[bb3]]:
  ; CHECK: llvm.br ^[[bb4]](%[[invoke]] : i32)
; CHECK: ^[[bb4]](%[[barg:.*]]: i32):
bb0:
  %ret = phi i32 [ 0, %nocall ], [ %invoke, %call ]
  ; CHECK: llvm.return %[[barg]] : i32
  ret i32 %ret
; CHECK: ^[[bb5]]:
bb1:
  ; CHECK: %[[lp:.*]] = llvm.landingpad cleanup : i32
  %resume = landingpad i32 cleanup
  ; CHECK: llvm.resume %[[lp]] : i32
  resume i32 %resume
}

; CHECK-LABEL: @invokePhiComplex
; CHECK-SAME:                   (%[[cond:.*]]: i1) -> i32
define i32 @invokePhiComplex(i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK: %[[c2:.*]] = llvm.mlir.constant(2 : i32) : i32
  ; CHECK: %[[c20:.*]] = llvm.mlir.constant(20 : i32) : i32  
  ; CHECK: llvm.cond_br %[[cond]], ^[[bb1:.*]], ^[[bb2:.*]]
  br i1 %cond, label %call, label %nocall
; CHECK: ^[[bb1]]:
call:
  ; CHECK: %[[invoke:.*]] = llvm.invoke @foo2() to ^[[bb3:.*]] unwind ^[[bb5:.*]] : () -> i32
  %invoke = invoke i32 @foo2() to label %bb0 unwind label %bb1
; CHECK: ^[[bb2]]:
nocall:
  ; CHECK: llvm.br ^[[bb4:.*]](%[[c0]], %[[c1]], %[[c2]] : i32, i32, i32)
  br label %bb0
; CHECK: ^[[bb3]]:
  ; CHECK: llvm.br ^[[bb4]](%[[invoke]], %[[c20]], %[[invoke]] : i32, i32, i32)
; CHECK: ^[[bb4]](%[[barg0:.*]]: i32, %[[barg1:.*]]: i32, %[[barg2:.*]]: i32):
bb0:
  %a = phi i32 [ 0, %nocall ], [ %invoke, %call ]
  %b = phi i32 [ 1, %nocall ], [ 20, %call ]
  %c = phi i32 [ 2, %nocall ], [ %invoke, %call ]
  ; CHECK: %[[add0:.*]] = llvm.add %[[barg0]], %[[barg1]] : i32
  ; CHECK: %[[add1:.*]] = llvm.add %[[barg2]], %[[add0]] : i32
  %d = add i32 %a, %b
  %e = add i32 %c, %d
  ; CHECK: llvm.return %[[add1]] : i32
  ret i32 %e
; CHECK: ^[[bb5]]:
bb1:
  ; CHECK: %[[lp:.*]] = llvm.landingpad cleanup : i32
  %resume = landingpad i32 cleanup
  ; CHECK: llvm.resume %[[lp]] : i32
  resume i32 %resume
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
%struct.va_list = type { ptr }

declare void @llvm.va_start(ptr)
declare void @llvm.va_copy(ptr, ptr)
declare void @llvm.va_end(ptr)

; CHECK-LABEL: llvm.func @variadic_function
define void @variadic_function(i32 %X, ...) {
  ; CHECK: %[[ALLOCA0:.+]] = llvm.alloca %{{.*}} x !llvm.struct<"struct.va_list", (ptr)> {{.*}} : (i32) -> !llvm.ptr
  %ap = alloca %struct.va_list
  ; CHECK: llvm.intr.vastart %[[ALLOCA0]]
  call void @llvm.va_start(ptr %ap)

  ; CHECK: %[[ALLOCA1:.+]] = llvm.alloca %{{.*}} x !llvm.ptr {{.*}} : (i32) -> !llvm.ptr
  %aq = alloca ptr
  ; CHECK: llvm.intr.vacopy %[[ALLOCA0]] to %[[ALLOCA1]]
  call void @llvm.va_copy(ptr %aq, ptr %ap)
  ; CHECK: llvm.intr.vaend %[[ALLOCA1]]
  call void @llvm.va_end(ptr %aq)

  ; CHECK: llvm.intr.vaend %[[ALLOCA0]]
  call void @llvm.va_end(ptr %ap)
  ; CHECK: llvm.return
  ret void
}
