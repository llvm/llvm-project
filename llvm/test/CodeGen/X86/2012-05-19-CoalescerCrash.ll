; RUN: llc < %s -verify-coalescing
; PR12892
;
; Dead code elimination during coalesing causes a live range to split into two
; virtual registers. Stale identity copies that had already been joined were
; interfering with the liveness computations.

target triple = "i386-pc-linux-gnu"

define void @_ZN4llvm17AsmMatcherEmitter3runERNS_11raw_ostreamE() align 2 personality ptr @__gxx_personality_v0 {
  invoke void @_ZNK4llvm13CodeGenTarget12getAsmParserEv()
          to label %1 unwind label %5

; <label>:1                                       ; preds = %0
  invoke void @_ZNK4llvm6Record16getValueAsStringENS_9StringRefE()
          to label %4 unwind label %2

; <label>:2                                       ; preds = %1
  %3 = landingpad { ptr, i32 }
          cleanup
  unreachable

; <label>:4                                       ; preds = %1
  invoke void @_ZN4llvm18isCurrentDebugTypeEPKc()
          to label %12 unwind label %7

; <label>:5                                       ; preds = %0
  %6 = landingpad { ptr, i32 }
          cleanup
  br label %33

; <label>:7                                       ; preds = %4
  %8 = landingpad { ptr, i32 }
          cleanup
  br label %9

; <label>:9                                       ; preds = %28, %7
  %10 = phi { ptr, i32 } [ %29, %28 ], [ %8, %7 ]
  %11 = extractvalue { ptr, i32 } %10, 1
  invoke fastcc void @_ZN12_GLOBAL__N_114AsmMatcherInfoD2Ev()
          to label %32 unwind label %35

; <label>:12                                      ; preds = %4
  invoke void @_ZNK4llvm13CodeGenTarget10getRegBankEv()
          to label %13 unwind label %16

; <label>:13                                      ; preds = %12
  br label %14

; <label>:14                                      ; preds = %20, %13
  %15 = icmp eq i32 undef, 0
  br i1 %15, label %20, label %18

; <label>:16                                      ; preds = %12
  %17 = landingpad { ptr, i32 }
          cleanup
  br label %26

; <label>:18                                      ; preds = %14
  invoke void @_ZNSs4_Rep9_S_createEjjRKSaIcE()
          to label %19 unwind label %21

; <label>:19                                      ; preds = %18
  unreachable

; <label>:20                                      ; preds = %14
  br label %14

; <label>:21                                      ; preds = %18
  %22 = landingpad { ptr, i32 }
          cleanup
  %23 = extractvalue { ptr, i32 } %22, 1
  br i1 undef, label %26, label %24

; <label>:24                                      ; preds = %21
  br i1 undef, label %25, label %26

; <label>:25                                      ; preds = %24
  unreachable

; <label>:26                                      ; preds = %24, %21, %16
  %27 = phi i32 [ 0, %16 ], [ %23, %21 ], [ %23, %24 ]
  invoke void @_ZNSt6vectorISt4pairISsSsESaIS1_EED1Ev()
          to label %28 unwind label %30

; <label>:28                                      ; preds = %26
  %29 = insertvalue { ptr, i32 } undef, i32 %27, 1
  br label %9

; <label>:30                                      ; preds = %26
  %31 = landingpad { ptr, i32 }
          catch ptr null
  unreachable

; <label>:32                                      ; preds = %9
  br label %33

; <label>:33                                      ; preds = %32, %5
  %34 = phi i32 [ undef, %5 ], [ %11, %32 ]
  unreachable

; <label>:35                                      ; preds = %9
  %36 = landingpad { ptr, i32 }
          catch ptr null
  unreachable
}

declare void @_ZNK4llvm13CodeGenTarget12getAsmParserEv()

declare i32 @__gxx_personality_v0(...)

declare void @_ZNK4llvm6Record16getValueAsStringENS_9StringRefE()

declare void @_ZN4llvm18isCurrentDebugTypeEPKc()

declare fastcc void @_ZN12_GLOBAL__N_114AsmMatcherInfoD2Ev() unnamed_addr inlinehint align 2

declare hidden void @_ZNSt6vectorISt4pairISsSsESaIS1_EED1Ev() unnamed_addr align 2

declare void @_ZNSs4_Rep9_S_createEjjRKSaIcE()

declare void @_ZNK4llvm13CodeGenTarget10getRegBankEv()
