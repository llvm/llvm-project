; RUN: mlir-translate -import-llvm %s | FileCheck %s
; RUN: mlir-translate -import-llvm -mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-DBG

; CHECK-DBG: #[[UNKNOWNLOC:.+]] = loc(unknown)

%struct.t = type {}
%struct.s = type { %struct.t, i64 }

; CHECK: llvm.mlir.global external @g1() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.struct<"struct.s", (struct<"struct.t", ()>, i64)>
@g1 = external global %struct.s, align 8
; CHECK: llvm.mlir.global external @g2() {addr_space = 0 : i32, alignment = 8 : i64} : f64
@g2 = external global double, align 8
; CHECK: llvm.mlir.global internal @g3("string")
@g3 = internal global [6 x i8] c"string"

; CHECK: llvm.mlir.global external @g5() {addr_space = 0 : i32} : vector<8xi32>
@g5 = external global <8 x i32>

; CHECK: llvm.mlir.global private @alig32(42 : i64) {addr_space = 0 : i32, alignment = 32 : i64, dso_local} : i64
@alig32 = private global i64 42, align 32

; CHECK: llvm.mlir.global private @alig64(42 : i64) {addr_space = 0 : i32, alignment = 64 : i64, dso_local} : i64
@alig64 = private global i64 42, align 64

@g4 = external global i32, align 8
; CHECK: llvm.mlir.global internal constant @int_gep() {addr_space = 0 : i32, dso_local} : !llvm.ptr<i32> {
; CHECK-DAG:   %[[addr:[0-9]+]] = llvm.mlir.addressof @g4 : !llvm.ptr<i32>
; CHECK-DAG:   %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK-NEXT:  %[[gepinit:[0-9]+]] = llvm.getelementptr %[[addr]][%[[c2]]] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
; CHECK-NEXT:  llvm.return %[[gepinit]] : !llvm.ptr<i32>
; CHECK-NEXT: }
@int_gep = internal constant i32* getelementptr (i32, i32* @g4, i32 2)

;
; dso_local attribute
;

; CHECK: llvm.mlir.global external @dso_local_var() {addr_space = 0 : i32, dso_local} : !llvm.struct<"struct.s", (struct<"struct.t", ()>, i64)>
@dso_local_var = external dso_local global %struct.s

;
; thread_local attribute
;

; CHECK: llvm.mlir.global external thread_local @thread_local_var() {addr_space = 0 : i32} : !llvm.struct<"struct.s", (struct<"struct.t", ()>, i64)>
@thread_local_var = external thread_local global %struct.s

;
; addr_space attribute
;

; CHECK: llvm.mlir.global external @addr_space_var(0 : i32) {addr_space = 6 : i32} : i32
@addr_space_var = addrspace(6) global i32 0

;
; Linkage attribute.
;

; CHECK: llvm.mlir.global private @private(42 : i32) {addr_space = 0 : i32, dso_local} : i32
@private = private global i32 42
; CHECK: llvm.mlir.global internal @internal(42 : i32) {addr_space = 0 : i32, dso_local} : i32
@internal = internal global i32 42
; CHECK: llvm.mlir.global available_externally @available_externally(42 : i32) {addr_space = 0 : i32}  : i32
@available_externally = available_externally global i32 42
; CHECK: llvm.mlir.global linkonce @linkonce(42 : i32) {addr_space = 0 : i32} : i32
@linkonce = linkonce global i32 42
; CHECK: llvm.mlir.global weak @weak(42 : i32) {addr_space = 0 : i32} : i32
@weak = weak global i32 42
; CHECK: llvm.mlir.global common @common(0 : i32) {addr_space = 0 : i32} : i32
@common = common global i32 zeroinitializer
; CHECK: llvm.mlir.global appending @appending(dense<[0, 1]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>
@appending = appending global [2 x i32] [i32 0, i32 1]
; CHECK: llvm.mlir.global extern_weak @extern_weak() {addr_space = 0 : i32} : i32
@extern_weak = extern_weak global i32
; CHECK: llvm.mlir.global linkonce_odr @linkonce_odr(42 : i32) {addr_space = 0 : i32} : i32
@linkonce_odr = linkonce_odr global i32 42
; CHECK: llvm.mlir.global weak_odr @weak_odr(42 : i32) {addr_space = 0 : i32} : i32
@weak_odr = weak_odr global i32 42
; CHECK: llvm.mlir.global external @external() {addr_space = 0 : i32} : i32
@external = external global i32

;
; UnnamedAddr attribute.
;


; CHECK: llvm.mlir.global private constant @no_unnamed_addr(42 : i64) {addr_space = 0 : i32, dso_local} : i64
@no_unnamed_addr = private constant i64 42
; CHECK: llvm.mlir.global private local_unnamed_addr constant @local_unnamed_addr(42 : i64) {addr_space = 0 : i32, dso_local} : i64
@local_unnamed_addr = private local_unnamed_addr constant i64 42
; CHECK: llvm.mlir.global private unnamed_addr constant @unnamed_addr(42 : i64) {addr_space = 0 : i32, dso_local} : i64
@unnamed_addr = private unnamed_addr constant i64 42

;
; Section attribute
;

; CHECK: llvm.mlir.global internal constant @sectionvar("teststring") {addr_space = 0 : i32, dso_local, section = ".mysection"}
@sectionvar = internal constant [10 x i8] c"teststring", section ".mysection"

;
; Sequential constants.
;

; CHECK: llvm.mlir.global internal constant @vector_constant(dense<[1, 2]> : vector<2xi32>) {addr_space = 0 : i32, dso_local} : vector<2xi32>
@vector_constant = internal constant <2 x i32> <i32 1, i32 2>
; CHECK: llvm.mlir.global internal constant @array_constant(dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>) {addr_space = 0 : i32, dso_local} : !llvm.array<2 x f32>
@array_constant = internal constant [2 x float] [float 1., float 2.]
; CHECK: llvm.mlir.global internal constant @nested_array_constant(dense<[{{\[}}1, 2], [3, 4]]> : tensor<2x2xi32>) {addr_space = 0 : i32, dso_local} : !llvm.array<2 x array<2 x i32>>
@nested_array_constant = internal constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]
; CHECK: llvm.mlir.global internal constant @nested_array_constant3(dense<[{{\[}}[1, 2], [3, 4]]]> : tensor<1x2x2xi32>) {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x array<2 x i32>>>
@nested_array_constant3 = internal constant [1 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]]
; CHECK: llvm.mlir.global internal constant @nested_array_vector(dense<[{{\[}}[1, 2], [3, 4]]]> : vector<1x2x2xi32>) {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x vector<2xi32>>>
@nested_array_vector = internal constant [1 x [2 x <2 x i32>]] [[2 x <2 x i32>] [<2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>]]

;
; Linkage on functions.
;

; CHECK: llvm.func internal @func_internal
define internal void @func_internal() {
  ret void
}

; CHECK: llvm.func @fe(i32) -> f32
declare float @fe(i32)

; CHECK: llvm.func internal spir_funccc @spir_func_internal()
define internal spir_func void @spir_func_internal() {
  ret void
}

; FIXME: function attributes.
; CHECK-LABEL: llvm.func internal @f1(%arg0: i64) -> i32 attributes {dso_local} {
; CHECK-DBG: llvm.func internal @f1(%arg0: i64 loc(unknown)) -> i32 attributes {dso_local} {
; CHECK-DAG: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK-DAG: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
; CHECK-DAG: %[[c1:[0-9]+]] = llvm.mlir.constant(true) : i1
; CHECK-DAG: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
  %aa = inttoptr i64 %a to i64*
; CHECK-DBG: llvm.mlir.addressof @g2 : !llvm.ptr<f64> loc(#[[UNKNOWNLOC]])
; %[[addrof:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm.ptr<f64>
; %[[addrof2:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm.ptr<f64>
; %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
; %{{[0-9]+}} = llvm.ptrtoint %[[addrof2]] : !llvm.ptr<f64> to i64
; %{{[0-9]+}} = llvm.getelementptr %[[addrof]][%3] : (!llvm.ptr<f64>, i32) -> !llvm.ptr<f64>
  %bb = ptrtoint double* @g2 to i64
  %cc = getelementptr double, double* @g2, i32 2
; CHECK: %[[b:[0-9]+]] = llvm.trunc %arg0 : i64 to i32
; CHECK-DBG: llvm.trunc %arg0 : i64 to i32 loc(#[[UNKNOWNLOC]])
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
; CHECK-DBG: } loc(#[[UNKNOWNLOC]])

; Test that instructions that dominate can be out of sequential order.
; CHECK-LABEL: llvm.func @f2(%arg0: i64) -> i64 {
; CHECK-DAG: %[[c3:[0-9]+]] = llvm.mlir.constant(3 : i64) : i64
define i64 @f2(i64 %a) noduplicate {
entry:
; CHECK: llvm.br ^bb2
  br label %next

; CHECK: ^bb1:
end:
; CHECK: llvm.return %1
  ret i64 %b

; CHECK: ^bb2:
next:
; CHECK: %1 = llvm.add %arg0, %[[c3]] : i64
  %b = add i64 %a, 3
; CHECK: llvm.br ^bb1
  br label %end
}

; Test arguments/phis.
; CHECK-LABEL: llvm.func @f2_phis(%arg0: i64) -> i64 {
; CHECK-DAG: %[[c3:[0-9]+]] = llvm.mlir.constant(3 : i64) : i64
define i64 @f2_phis(i64 %a) noduplicate {
entry:
; CHECK: llvm.br ^bb2
  br label %next

; CHECK: ^bb1(%1: i64):
end:
  %c = phi i64 [ %b, %next ]
; CHECK: llvm.return %1
  ret i64 %c

; CHECK: ^bb2:
next:
; CHECK: %2 = llvm.add %arg0, %[[c3]] : i64
  %b = add i64 %a, 3
; CHECK: llvm.br ^bb1
  br label %end
}

; CHECK-LABEL: llvm.func @f3() -> !llvm.ptr<i32>
define i32* @f3() {
; CHECK: %[[c:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm.ptr<f64>
; CHECK: %[[b:[0-9]+]] = llvm.bitcast %[[c]] : !llvm.ptr<f64> to !llvm.ptr<i32>
; CHECK: llvm.return %[[b]] : !llvm.ptr<i32>
  ret i32* bitcast (double* @g2 to i32*)
}

; CHECK-LABEL: llvm.func @f6(%arg0: !llvm.ptr<func<void (i16)>>)
define void @f6(void (i16) *%fn) {
; CHECK: %[[c:[0-9]+]] = llvm.mlir.constant(0 : i16) : i16
; CHECK: llvm.call %arg0(%[[c]])
  call void %fn(i16 0)
  ret void
}

; Testing rest of the floating point constant kinds.
; CHECK-LABEL: llvm.func @FPConstant(%arg0: f16, %arg1: bf16, %arg2: f128, %arg3: f80)
define void @FPConstant(half %a, bfloat %b, fp128 %c, x86_fp80 %d) {
  ; CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(7.000000e+00 : f80) : f80
  ; CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(0.000000e+00 : f128) : f128
  ; CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(1.000000e+00 : bf16) : bf16
  ; CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16

  ; CHECK: llvm.fadd %[[C3]], %arg0  : f16
  %1 = fadd half 1.0, %a
  ; CHECK: llvm.fadd %[[C2]], %arg1  : bf16
  %2 = fadd bfloat 1.0, %b
  ; CHECK: llvm.fadd %[[C1]], %arg2  : f128
  %3 = fadd fp128 0xL00000000000000000000000000000000, %c
  ; CHECK: llvm.fadd %[[C0]], %arg3  : f80
  %4 = fadd x86_fp80 0xK4001E000000000000000, %d
  ret void
}

;
; Functions as constants.
;

; Calling the function that has not been defined yet.
; CHECK-LABEL: @precaller
define i32 @precaller() {
  %1 = alloca i32 ()*
  ; CHECK: %[[func:.*]] = llvm.mlir.addressof @callee : !llvm.ptr<func<i32 ()>>
  ; CHECK: llvm.store %[[func]], %[[loc:.*]]
  store i32 ()* @callee, i32 ()** %1
  ; CHECK: %[[indir:.*]] = llvm.load %[[loc]]
  %2 = load i32 ()*, i32 ()** %1
  ; CHECK: llvm.call %[[indir]]()
  %3 = call i32 %2()
  ret i32 %3
}

define i32 @callee() {
  ret i32 42
}

; Calling the function that has been defined.
; CHECK-LABEL: @postcaller
define i32 @postcaller() {
  %1 = alloca i32 ()*
  ; CHECK: %[[func:.*]] = llvm.mlir.addressof @callee : !llvm.ptr<func<i32 ()>>
  ; CHECK: llvm.store %[[func]], %[[loc:.*]]
  store i32 ()* @callee, i32 ()** %1
  ; CHECK: %[[indir:.*]] = llvm.load %[[loc]]
  %2 = load i32 ()*, i32 ()** %1
  ; CHECK: llvm.call %[[indir]]()
  %3 = call i32 %2()
  ret i32 %3
}

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

; Switch instruction
declare void @g(i32)

; CHECK-LABEL: llvm.func @simple_switch(%arg0: i32) {
define void @simple_switch(i32 %val) {
; CHECK: %[[C0:.+]] = llvm.mlir.constant(11 : i32) : i32
; CHECK: %[[C1:.+]] = llvm.mlir.constant(87 : i32) : i32
; CHECK: %[[C2:.+]] = llvm.mlir.constant(78 : i32) : i32
; CHECK: %[[C3:.+]] = llvm.mlir.constant(94 : i32) : i32
; CHECK: %[[C4:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK: llvm.switch %arg0 : i32, ^[[BB5:.+]] [
; CHECK:   0: ^[[BB1:.+]],
; CHECK:   9: ^[[BB2:.+]],
; CHECK:   994: ^[[BB3:.+]],
; CHECK:   1154: ^[[BB4:.+]]
; CHECK: ]
  switch i32 %val, label %def [
    i32 0, label %one
    i32 9, label %two
    i32 994, label %three
    i32 1154, label %four
  ]

; CHECK: ^[[BB1]]:
; CHECK: llvm.call @g(%[[C4]]) : (i32) -> ()
; CHECK: llvm.return
one:
  call void @g(i32 1)
  ret void
; CHECK: ^[[BB2]]:
; CHECK: llvm.call @g(%[[C3]]) : (i32) -> ()
; CHECK: llvm.return
two:
  call void @g(i32 94)
  ret void
; CHECK: ^[[BB3]]:
; CHECK: llvm.call @g(%[[C2]]) : (i32) -> ()
; CHECK: llvm.return
three:
  call void @g(i32 78)
  ret void
; CHECK: ^[[BB4]]:
; CHECK: llvm.call @g(%[[C1]]) : (i32) -> ()
; CHECK: llvm.return
four:
  call void @g(i32 87)
  ret void
; CHECK: ^[[BB5]]:
; CHECK: llvm.call @g(%[[C0]]) : (i32) -> ()
; CHECK: llvm.return
def:
  call void @g(i32 11)
  ret void
}

; CHECK-LABEL: llvm.func @switch_args(%arg0: i32) {
define void @switch_args(i32 %val) {
  ; CHECK: %[[C0:.+]] = llvm.mlir.constant(44 : i32) : i32
  ; CHECK: %[[C1:.+]] = llvm.mlir.constant(34 : i32) : i32
  ; CHECK: %[[C2:.+]] = llvm.mlir.constant(33 : i32) : i32
  %pred = icmp ult i32 %val, 87
  br i1 %pred, label %bbs, label %bb1

bb1:
  %vx = add i32 %val, 22
  %pred2 = icmp ult i32 %val, 94
  br i1 %pred2, label %bb2, label %bb3

bb2:
  %vx0 = add i32 %val, 23
  br label %one

bb3:
  br label %def

; CHECK: %[[V1:.+]] = llvm.add %arg0, %[[C2]] : i32
; CHECK: %[[V2:.+]] = llvm.add %arg0, %[[C1]] : i32
; CHECK: %[[V3:.+]] = llvm.add %arg0, %[[C0]] : i32
; CHECK: llvm.switch %arg0 : i32, ^[[BBD:.+]](%[[V3]] : i32) [
; CHECK:   0: ^[[BB1:.+]](%[[V1]], %[[V2]] : i32, i32)
; CHECK: ]
bbs:
  %vy = add i32 %val, 33
  %vy0 = add i32 %val, 34
  %vz = add i32 %val, 44
  switch i32 %val, label %def [
    i32 0, label %one
  ]

; CHECK: ^[[BB1]](%[[BA0:.+]]: i32, %[[BA1:.+]]: i32):
one: ; pred: bb2, bbs
  %v0 = phi i32 [%vx, %bb2], [%vy, %bbs]
  %v1 = phi i32 [%vx0, %bb2], [%vy0, %bbs]
  ; CHECK: llvm.add %[[BA0]], %[[BA1]]  : i32
  %vf = add i32 %v0, %v1
  call void @g(i32 %vf)
  ret void

; CHECK: ^[[BBD]](%[[BA2:.+]]: i32):
def: ; pred: bb3, bbs
  %v2 = phi i32 [%vx, %bb3], [%vz, %bbs]
  ; CHECK: llvm.call @g(%[[BA2]])
  call void @g(i32 %v2)
  ret void
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
