; RUN: mlir-translate -import-llvm %s | FileCheck %s

%struct.t = type {}
%struct.s = type { %struct.t, i64 }

; CHECK: llvm.mlir.global external @g1() {alignment = 8 : i64} : !llvm.struct<"struct.s", (struct<"struct.t", ()>, i64)>
@g1 = external global %struct.s, align 8
; CHECK: llvm.mlir.global external @g2() {alignment = 8 : i64} : f64
@g2 = external global double, align 8
; CHECK: llvm.mlir.global internal @g3("string")
@g3 = internal global [6 x i8] c"string"

; CHECK: llvm.mlir.global external @g5() : vector<8xi32>
@g5 = external global <8 x i32>

; CHECK: llvm.mlir.global private @alig32(42 : i64) {alignment = 32 : i64} : i64
@alig32 = private global i64 42, align 32

; CHECK: llvm.mlir.global private @alig64(42 : i64) {alignment = 64 : i64} : i64
@alig64 = private global i64 42, align 64

@g4 = external global i32, align 8
; CHECK: llvm.mlir.global internal constant @int_gep() : !llvm.ptr<i32> {
; CHECK-DAG:   %[[addr:[0-9]+]] = llvm.mlir.addressof @g4 : !llvm.ptr<i32>
; CHECK-DAG:   %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK-NEXT:  %[[gepinit:[0-9]+]] = llvm.getelementptr %[[addr]][%[[c2]]] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
; CHECK-NEXT:  llvm.return %[[gepinit]] : !llvm.ptr<i32>
; CHECK-NEXT: }
@int_gep = internal constant i32* getelementptr (i32, i32* @g4, i32 2)

;
; Linkage attribute.
;

; CHECK: llvm.mlir.global private @private(42 : i32) : i32
@private = private global i32 42
; CHECK: llvm.mlir.global internal @internal(42 : i32) : i32
@internal = internal global i32 42
; CHECK: llvm.mlir.global available_externally @available_externally(42 : i32) : i32
@available_externally = available_externally global i32 42
; CHECK: llvm.mlir.global linkonce @linkonce(42 : i32) : i32
@linkonce = linkonce global i32 42
; CHECK: llvm.mlir.global weak @weak(42 : i32) : i32
@weak = weak global i32 42
; CHECK: llvm.mlir.global common @common(0 : i32) : i32
@common = common global i32 zeroinitializer
; CHECK: llvm.mlir.global appending @appending(dense<[0, 1]> : tensor<2xi32>) : !llvm.array<2 x i32>
@appending = appending global [2 x i32] [i32 0, i32 1]
; CHECK: llvm.mlir.global extern_weak @extern_weak() : i32
@extern_weak = extern_weak global i32
; CHECK: llvm.mlir.global linkonce_odr @linkonce_odr(42 : i32) : i32
@linkonce_odr = linkonce_odr global i32 42
; CHECK: llvm.mlir.global weak_odr @weak_odr(42 : i32) : i32
@weak_odr = weak_odr global i32 42
; CHECK: llvm.mlir.global external @external() : i32
@external = external global i32

;
; UnnamedAddr attribute.
;


; CHECK: llvm.mlir.global private constant @no_unnamed_addr(42 : i64) : i64
@no_unnamed_addr = private constant i64 42
; CHECK: llvm.mlir.global private local_unnamed_addr constant @local_unnamed_addr(42 : i64) : i64
@local_unnamed_addr = private local_unnamed_addr constant i64 42
; CHECK: llvm.mlir.global private unnamed_addr constant @unnamed_addr(42 : i64) : i64
@unnamed_addr = private unnamed_addr constant i64 42

;
; Section attribute
;

; CHECK: llvm.mlir.global internal constant @sectionvar("teststring")  {section = ".mysection"}
@sectionvar = internal constant [10 x i8] c"teststring", section ".mysection"

;
; Sequential constants.
;

; CHECK: llvm.mlir.global internal constant @vector_constant(dense<[1, 2]> : vector<2xi32>) : vector<2xi32>
@vector_constant = internal constant <2 x i32> <i32 1, i32 2>
; CHECK: llvm.mlir.global internal constant @array_constant(dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>) : !llvm.array<2 x f32>
@array_constant = internal constant [2 x float] [float 1., float 2.]
; CHECK: llvm.mlir.global internal constant @nested_array_constant(dense<[{{\[}}1, 2], [3, 4]]> : tensor<2x2xi32>) : !llvm.array<2 x array<2 x i32>>
@nested_array_constant = internal constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]
; CHECK: llvm.mlir.global internal constant @nested_array_constant3(dense<[{{\[}}[1, 2], [3, 4]]]> : tensor<1x2x2xi32>) : !llvm.array<1 x array<2 x array<2 x i32>>>
@nested_array_constant3 = internal constant [1 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]]
; CHECK: llvm.mlir.global internal constant @nested_array_vector(dense<[{{\[}}[1, 2], [3, 4]]]> : vector<1x2x2xi32>) : !llvm.array<1 x array<2 x vector<2xi32>>>
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

; FIXME: function attributes.
; CHECK-LABEL: llvm.func internal @f1(%arg0: i64) -> i32 {
; CHECK-DAG: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK-DAG: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
; CHECK-DAG: %[[c1:[0-9]+]] = llvm.mlir.constant(true) : i1
; CHECK-DAG: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
  %aa = inttoptr i64 %a to i64*
; %[[addrof:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm.ptr<f64>
; %[[addrof2:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm.ptr<f64>
; %{{[0-9]+}} = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i64>
; %{{[0-9]+}} = llvm.ptrtoint %[[addrof2]] : !llvm.ptr<f64> to i64
; %{{[0-9]+}} = llvm.getelementptr %[[addrof]][%3] : (!llvm.ptr<f64>, i32) -> !llvm.ptr<f64>
  %bb = ptrtoint double* @g2 to i64
  %cc = getelementptr double, double* @g2, i32 2
; CHECK: %[[b:[0-9]+]] = llvm.trunc %arg0 : i64 to i32
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

; CHECK-LABEL: llvm.func @f4() -> !llvm.ptr<i32>
define i32* @f4() {
; CHECK: %[[b:[0-9]+]] = llvm.mlir.null : !llvm.ptr<i32>
; CHECK: llvm.return %[[b]] : !llvm.ptr<i32>
  ret i32* bitcast (double* null to i32*)
}

; CHECK-LABEL: llvm.func @f5
define void @f5(i32 %d) {
; FIXME: icmp should return i1.
; CHECK: = llvm.icmp "eq"
  %1 = icmp eq i32 %d, 2
; CHECK: = llvm.icmp "slt"
  %2 = icmp slt i32 %d, 2
; CHECK: = llvm.icmp "sle"
  %3 = icmp sle i32 %d, 2
; CHECK: = llvm.icmp "sgt"
  %4 = icmp sgt i32 %d, 2
; CHECK: = llvm.icmp "sge"
  %5 = icmp sge i32 %d, 2
; CHECK: = llvm.icmp "ult"
  %6 = icmp ult i32 %d, 2
; CHECK: = llvm.icmp "ule"
  %7 = icmp ule i32 %d, 2
; CHECK: = llvm.icmp "ugt"
  %8 = icmp ugt i32 %d, 2
  ret void
}

; CHECK-LABEL: llvm.func @f6(%arg0: !llvm.ptr<func<void (i16)>>)
define void @f6(void (i16) *%fn) {
; CHECK: %[[c:[0-9]+]] = llvm.mlir.constant(0 : i16) : i16
; CHECK: llvm.call %arg0(%[[c]])
  call void %fn(i16 0)
  ret void
}

; CHECK-LABEL: llvm.func @FPArithmetic(%arg0: f32, %arg1: f32, %arg2: f64, %arg3: f64)
define void @FPArithmetic(float %a, float %b, double %c, double %d) {
  ; CHECK: %[[a1:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f64) : f64
  ; CHECK: %[[a2:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f32) : f32
  ; CHECK: %[[a3:[0-9]+]] = llvm.fadd %[[a2]], %arg0 : f32
  %1 = fadd float 0x403E4CCCC0000000, %a
  ; CHECK: %[[a4:[0-9]+]] = llvm.fadd %arg0, %arg1 : f32
  %2 = fadd float %a, %b
  ; CHECK: %[[a5:[0-9]+]] = llvm.fadd %[[a1]], %arg2 : f64
  %3 = fadd double 3.030000e+01, %c
  ; CHECK: %[[a6:[0-9]+]] = llvm.fsub %arg0, %arg1 : f32
  %4 = fsub float %a, %b
  ; CHECK: %[[a7:[0-9]+]] = llvm.fsub %arg2, %arg3 : f64
  %5 = fsub double %c, %d
  ; CHECK: %[[a8:[0-9]+]] = llvm.fmul %arg0, %arg1 : f32
  %6 = fmul float %a, %b
  ; CHECK: %[[a9:[0-9]+]] = llvm.fmul %arg2, %arg3 : f64
  %7 = fmul double %c, %d
  ; CHECK: %[[a10:[0-9]+]] = llvm.fdiv %arg0, %arg1 : f32
  %8 = fdiv float %a, %b
  ; CHECK: %[[a12:[0-9]+]] = llvm.fdiv %arg2, %arg3 : f64
  %9 = fdiv double %c, %d
  ; CHECK: %[[a11:[0-9]+]] = llvm.frem %arg0, %arg1 : f32
  %10 = frem float %a, %b
  ; CHECK: %[[a13:[0-9]+]] = llvm.frem %arg2, %arg3 : f64
  %11 = frem double %c, %d
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
  ; CHECK: %[[a3:[0-9]+]] = llvm.alloca %{{[0-9]+}} x i8 : (i32) -> !llvm.ptr<i8>
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

;CHECK-LABEL: @useFenceInst
define i32 @useFenceInst() {
  ;CHECK: llvm.fence syncscope("agent") seq_cst
  fence syncscope("agent") seq_cst
  ;CHECK: llvm.fence release
  fence release
  ;CHECK: llvm.fence seq_cst
  fence syncscope("") seq_cst
  ret i32 0
}
