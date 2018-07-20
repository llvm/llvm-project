; RUN: llc -o - %s | FileCheck %s
; Make sure RegAllocGreedy/SplitKit do not produce invalid liveness information
; and crash when splitting a liverange twice and rematerializing each time.
; (Sorry for the testcase; this was ran through bugpoint and then  manually
;  reduced for several hours but is still big...)
target triple = "thumbv7-apple-ios"

%struct.ham = type { %struct.wombat.0 }
%struct.wombat.0 = type { %struct.barney }
%struct.barney = type { %struct.snork.1 }
%struct.snork.1 = type { %struct.wobble.2 }
%struct.wobble.2 = type { %struct.blam }
%struct.blam = type { i32, i32, i8* }
%struct.ham.3 = type { %struct.pluto }
%struct.pluto = type { %struct.zot*, %struct.snork.5, %struct.wibble }
%struct.zot = type { %struct.blam.4* }
%struct.blam.4 = type <{ %struct.zot, %struct.blam.4*, %struct.zot*, i8, [3 x i8] }>
%struct.snork.5 = type { %struct.quux }
%struct.quux = type { %struct.zot }
%struct.wibble = type { %struct.widget }
%struct.widget = type { i32 }
%struct.bar = type { %struct.spam }
%struct.spam = type { %struct.zot*, %struct.wobble, %struct.zot.7 }
%struct.wobble = type { %struct.wibble.6 }
%struct.wibble.6 = type { %struct.zot }
%struct.zot.7 = type { %struct.ham.8 }
%struct.ham.8 = type { i32 }
%struct.hoge = type { %struct.ham, %struct.foo }
%struct.foo = type { float, float }
%struct.wombat = type { %struct.ham, float }
%struct.snork = type { %struct.ham.9, [11 x i8] }
%struct.ham.9 = type { i8 }

@global = external global i8
@global.1 = private constant [20 x i8] c"aaaaaaaaaaaaaaaaaa0\00"
@global.2 = external constant [27 x i8]
@global.3 = external global %struct.ham
@global.4 = external constant [47 x i8]
@global.5 = external constant [61 x i8]
@global.6 = external constant [40 x i8]
@global.7 = external constant [24 x i8]
@global.8 = external constant [20 x i8]
@global.9 = external global %struct.ham
@global.10 = external global %struct.ham
@global.11 = external global %struct.ham
@global.12 = external global %struct.ham
@global.13 = external global %struct.ham
@global.14 = external global %struct.ham
@global.15 = external global %struct.ham
@global.16 = external global %struct.ham
@global.17 = external global %struct.ham
@global.18 = external constant [35 x i8]
@global.19 = external global %struct.ham
@global.20 = external constant [53 x i8]
@global.21 = external global %struct.ham
@global.22 = external global %struct.ham
@global.23 = external global %struct.ham
@global.24 = external constant [32 x i8]
@global.25 = external global %struct.ham
@global.26 = external constant [47 x i8]
@global.27 = external global %struct.ham
@global.28 = external constant [45 x i8]
@global.29 = external global %struct.ham
@global.30 = external global %struct.ham
@global.31 = external constant [24 x i8]
@global.32 = external global %struct.ham
@global.33 = external global %struct.ham
@global.34 = external global %struct.ham
@global.35 = external global %struct.ham
@global.36 = external constant [27 x i8]
@global.37 = external global %struct.ham
@global.38 = external constant [10 x i8]
@global.39 = external global %struct.ham
@global.40 = external global %struct.ham
@global.41 = external global %struct.ham
@global.42 = external global %struct.ham
@global.43 = external global %struct.ham
@global.44 = external constant [41 x i8]
@global.45 = external global %struct.ham
@global.46 = external global %struct.ham
@global.47 = external global %struct.ham
@global.48 = external global %struct.ham
@global.49 = external constant [52 x i8]
@global.50 = external constant [47 x i8]
@global.51 = external global %struct.ham
@global.52 = external global %struct.ham
@global.53 = external global %struct.ham
@global.54 = external global %struct.ham
@global.55 = external global %struct.ham.3
@global.56 = external global %struct.bar
@global.57 = external global i8

declare %struct.ham* @bar(%struct.ham* returned)

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

declare %struct.ham* @wobble(%struct.ham* returned, %struct.ham* ) 

declare i32 @quux(...)

declare i8* @_Znwm(i32)

declare i32 @wobble.58(%struct.pluto*, [1 x i32], %struct.ham* , %struct.hoge* )

declare i32 @widget(%struct.spam*, [1 x i32], %struct.ham* , %struct.wombat* )

; Just check we didn't crash and did output something...
; CHECK-LABEL: func:
; CHECK: trap
define internal void @func() section "__TEXT,__StaticInit,regular,pure_instructions" personality i32 (...)* @quux {
  %tmp = tail call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.3 to i8*), i8* @global) #0
  %tmp2 = invoke %struct.ham* @wobble(%struct.ham* undef, %struct.ham*  @global.9)
          to label %bb14 unwind label %bbunwind

bb14:
  %tmp15 = getelementptr  i8, i8* undef, i32 12
  store i8 0, i8* %tmp15
  %tmp16 = icmp eq i8 undef, 0
  br i1 %tmp16, label %bb28, label %bb18

bb18:
  br i1 undef, label %bb21, label %bb29

bb21:
  %tmp22 = call i8* @_Znwm(i32 16)
  store i32 17, i32* getelementptr  (%struct.ham, %struct.ham* @global.10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp23 = call i8* @_Znwm(i32 32)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([27 x i8], [27 x i8]* @global.2, i32 0, i32 0), i32 26, i1 false)
  store i32 33, i32* getelementptr  (%struct.ham, %struct.ham* @global.11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store i32 23, i32* getelementptr  (%struct.ham, %struct.ham* @global.11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([24 x i8], [24 x i8]* @global.7, i32 0, i32 0), i32 23, i1 false)
  %tmp24 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.11 to i8*), i8* @global) #0
  store i32 49, i32* getelementptr  (%struct.ham, %struct.ham* @global.12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store i32 37, i32* getelementptr  (%struct.ham, %struct.ham* @global.13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.14 to i8*), i8 0, i32 12, i1 false)
  %tmp25 = call i8* @_Znwm(i32 48)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp25, i8* align 1 getelementptr  ([40 x i8], [40 x i8]* @global.6, i32 0, i32 0), i32 39, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([47 x i8], [47 x i8]* @global.4, i32 0, i32 0), i32 46, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([61 x i8], [61 x i8]* @global.5, i32 0, i32 0), i32 60, i1 false)
  %tmp26 = call i8* @_Znwm(i32 48)
  store i32 65, i32* getelementptr  (%struct.ham, %struct.ham* @global.15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp27 = icmp eq i8 undef, 0
  br i1 %tmp27, label %bb30, label %bb33

bb28:
  call void @llvm.trap()
  unreachable

bb29:
  call void @llvm.trap()
  unreachable

bb30:
  %tmp31 = icmp eq i32 undef, 37
  br i1 %tmp31, label %bb32, label %bb30

bb32:
  store i8 1, i8* @global.57
  br label %bb33

bb33:
  %tmp34 = call i8* @_Znwm(i32 32)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([20 x i8], [20 x i8]* @global.1, i32 0, i32 0), i32 19, i1 false)
  store i32 17, i32* getelementptr  (%struct.ham, %struct.ham* @global.16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store i32 65, i32* getelementptr  (%struct.ham, %struct.ham* @global.17, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([35 x i8], [35 x i8]* @global.18, i32 0, i32 0), i32 34, i1 false)
  store i32 65, i32* getelementptr  (%struct.ham, %struct.ham* @global.19, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([53 x i8], [53 x i8]* @global.20, i32 0, i32 0), i32 52, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([20 x i8], [20 x i8]* @global.8, i32 0, i32 0), i32 19, i1 false)
  store i32 37, i32* getelementptr  (%struct.ham, %struct.ham* @global.21, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  %tmp35 = call i8* @_Znwm(i32 32)
  store i8 16, i8* bitcast (%struct.ham* @global.22 to i8*)
  %tmp36 = call i8* @_Znwm(i32 32)
  store i32 31, i32* getelementptr  (%struct.ham, %struct.ham* @global.23, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp36, i8* align 1 getelementptr  ([32 x i8], [32 x i8]* @global.24, i32 0, i32 0), i32 31, i1 false)
  %tmp37 = getelementptr  i8, i8* %tmp36, i32 31
  store i8 0, i8* %tmp37
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([47 x i8], [47 x i8]* @global.26, i32 0, i32 0), i32 46, i1 false)
  %tmp38 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.25 to i8*), i8* @global) #0
  %tmp39 = call i8* @_Znwm(i32 48)
  store i32 44, i32* getelementptr  (%struct.ham, %struct.ham* @global.27, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp39, i8* align 1 getelementptr  ([45 x i8], [45 x i8]* @global.28, i32 0, i32 0), i32 44, i1 false)
  %tmp40 = getelementptr  i8, i8* %tmp39, i32 44
  store i8 0, i8* %tmp40
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.29 to i8*), i8 0, i32 12, i1 false)
  %tmp41 = call i8* @_Znwm(i32 32)
  store i32 23, i32* getelementptr  (%struct.ham, %struct.ham* @global.30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp41, i8* align 1 getelementptr  ([24 x i8], [24 x i8]* @global.31, i32 0, i32 0), i32 23, i1 false)
  %tmp42 = getelementptr  i8, i8* %tmp41, i32 23
  store i8 0, i8* %tmp42
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.32 to i8*), i8 0, i32 12, i1 false)
  store i8 16, i8* bitcast (%struct.ham* @global.32 to i8*)
  %tmp43 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.33 to i8*), i8* @global) #0
  %tmp44 = call i8* @_Znwm(i32 16)
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.34 to i8*), i8 0, i32 12, i1 false)
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.9 to i8*), i8 0, i32 12, i1 false)
  %tmp45 = call i8* @_Znwm(i32 32)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp45, i8* align 1 getelementptr  ([27 x i8], [27 x i8]* @global.36, i32 0, i32 0), i32 26, i1 false)
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.ham* @global.37 to i8*), i8 0, i32 12, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr (%struct.snork, %struct.snork* bitcast (%struct.ham* @global.37 to %struct.snork*), i32 0, i32 1, i32 0), i8* align 1 getelementptr  ([10 x i8], [10 x i8]* @global.38, i32 0, i32 0), i32 9, i1 false)
  store i32 17, i32* getelementptr  (%struct.ham, %struct.ham* @global.39, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp46 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.40 to i8*), i8* @global) #0
  %tmp47 = call i8* @_Znwm(i32 32)
  %tmp48 = getelementptr  i8, i8* %tmp47, i32 21
  store i8 0, i8* %tmp48
  store i32 33, i32* getelementptr  (%struct.ham, %struct.ham* @global.41, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store i32 15, i32* getelementptr  (%struct.ham, %struct.ham* @global.42, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  %tmp49 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.43 to i8*), i8* @global) #0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([41 x i8], [41 x i8]* @global.44, i32 0, i32 0), i32 40, i1 false)
  %tmp50 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.45 to i8*), i8* @global) #0
  %tmp51 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.46 to i8*), i8* @global) #0
  %tmp52 = call i8* @_Znwm(i32 32)
  store i8* %tmp52, i8** getelementptr  (%struct.ham, %struct.ham* @global.47, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([52 x i8], [52 x i8]* @global.49, i32 0, i32 0), i32 51, i1 false)
  %tmp53 = call i32 @__cxa_atexit(void (i8*)* bitcast (%struct.ham* (%struct.ham*)* @bar to void (i8*)*), i8* bitcast (%struct.ham* @global.48 to i8*), i8* @global) #0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 undef, i8* align 1 getelementptr  ([47 x i8], [47 x i8]* @global.50, i32 0, i32 0), i32 46, i1 false)
  store i32 33, i32* getelementptr  (%struct.ham, %struct.ham* @global.51, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store i32 37, i32* getelementptr  (%struct.ham, %struct.ham* @global.52, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  %tmp54 = invoke %struct.ham* @wobble(%struct.ham* undef, %struct.ham*  @global.54)
          to label %bb58 unwind label %bbunwind

bb58:
  %tmp59 = invoke i32 @wobble.58(%struct.pluto* getelementptr  (%struct.ham.3, %struct.ham.3* @global.55, i32 0, i32 0), [1 x i32] [i32 ptrtoint (%struct.zot* getelementptr  (%struct.ham.3, %struct.ham.3* @global.55, i32 0, i32 0, i32 1, i32 0, i32 0) to i32)], %struct.ham*  undef, %struct.hoge*  undef)
          to label %bb71 unwind label %bbunwind

bb71:
  %tmp72 = invoke i32 @widget(%struct.spam* getelementptr  (%struct.bar, %struct.bar* @global.56, i32 0, i32 0), [1 x i32] [i32 ptrtoint (%struct.zot* getelementptr  (%struct.bar, %struct.bar* @global.56, i32 0, i32 0, i32 1, i32 0, i32 0) to i32)], %struct.ham*  undef, %struct.wombat*  undef)
          to label %bb73 unwind label %bbunwind

bb73:
  ret void

bbunwind:
  %tmp75 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

declare void @llvm.trap()

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* , i8* , i32, i1)

declare void @llvm.memset.p0i8.i32(i8* , i8, i32, i1)

attributes #0 = { nounwind }
