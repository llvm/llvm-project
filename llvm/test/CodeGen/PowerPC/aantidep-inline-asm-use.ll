; RUN: llc -O2 < %s | FileCheck %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

; Function Attrs: nounwind
define void @_ZN10SubProcess19ScrubbedForkAndExecEiPiS0_PNS_7ResultsE() #0 align 2 {
; CHECK: lis 3, 1234
; CHECK-NOT: li 3
; CHECK-NOT: ori 3
; CHECK-NOT: addi 3
; CHECK-NOT: addis 3
; CHECK-NOT: lis 3
; CHECK: sc
  br i1 undef, label %1, label %2

; <label>:1                                       ; preds = %0
  br label %60

; <label>:2                                       ; preds = %0
  br i1 undef, label %3, label %4

; <label>:3                                       ; preds = %2
  unreachable

; <label>:4                                       ; preds = %2
  br i1 undef, label %.lr.ph111, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit

.lr.ph111:                                        ; preds = %4
  br label %5

_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit: ; preds = %12, %4
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader, label %13

; <label>:5                                       ; preds = %12, %.lr.ph111
  br i1 undef, label %6, label %9

; <label>:6                                       ; preds = %5
  br i1 undef, label %7, label %8

; <label>:7                                       ; preds = %6
  unreachable

; <label>:8                                       ; preds = %6
  br label %12

; <label>:9                                       ; preds = %5
  br i1 undef, label %10, label %11

; <label>:10                                      ; preds = %9
  br label %12

; <label>:11                                      ; preds = %9
  br label %12

; <label>:12                                      ; preds = %11, %10, %8
  br i1 undef, label %5, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit

; <label>:13                                      ; preds = %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader, label %14

; <label>:14                                      ; preds = %13
  br label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader

_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader: ; preds = %14, %13, %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls5closeEi.exit.preheader, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19._crit_edge

_ZN10SubProcess12SafeSyscalls5closeEi.exit.preheader: ; preds = %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19, %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader
  br label %_ZN10SubProcess12SafeSyscalls5closeEi.exit

_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19._crit_edge: ; preds = %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19, %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19.preheader
  br i1 undef, label %15, label %19

_ZN10SubProcess12SafeSyscalls5closeEi.exit:       ; preds = %_ZN10SubProcess12SafeSyscalls5closeEi.exit, %_ZN10SubProcess12SafeSyscalls5closeEi.exit.preheader
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls5closeEi.exit, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19

_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19: ; preds = %_ZN10SubProcess12SafeSyscalls5closeEi.exit
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19._crit_edge, label %_ZN10SubProcess12SafeSyscalls5closeEi.exit.preheader

; <label>:15                                      ; preds = %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19._crit_edge
  br label %16

; <label>:16                                      ; preds = %17, %15
  br i1 undef, label %17, label %.critedge.preheader

; <label>:17                                      ; preds = %16
  br i1 undef, label %16, label %.critedge.preheader

.critedge.preheader:                              ; preds = %17, %16
  br label %.critedge

.critedge:                                        ; preds = %18, %.critedge.preheader
  br i1 undef, label %18, label %.critedge8

; <label>:18                                      ; preds = %.critedge
  br i1 undef, label %.critedge, label %.critedge8

.critedge8:                                       ; preds = %18, %.critedge
  br label %59

; <label>:19                                      ; preds = %_ZN10SubProcess12SafeSyscalls11sigprocmaskEiPKNS0_15kernel_sigset_tEPS1_.exit19._crit_edge
  br label %_ZN10SubProcess12SafeSyscalls5closeEi.exit22

_ZN10SubProcess12SafeSyscalls5closeEi.exit22:     ; preds = %_ZN10SubProcess12SafeSyscalls5closeEi.exit22, %19
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls5closeEi.exit22, label %20

; <label>:20                                      ; preds = %_ZN10SubProcess12SafeSyscalls5closeEi.exit22
  %21 = alloca i8, i64 undef, align 1
  br label %.thread.outer

.thread.outer:                                    ; preds = %._crit_edge, %20
  br label %.thread

.thread:                                          ; preds = %45, %.thread.outer
  call void @llvm.memset.p0i8.i64(i8* undef, i8 0, i64 56, i32 8, i1 false)
  store i8* %21, i8** undef, align 8
  store i32 1073741824, i32* undef, align 8
  %22 = call { i64, i64, i64, i64, i64, i64, i64 } asm sideeffect "sc\0A\09mfcr $0", "=&{r0},=&{r3},=&{r4},=&{r5},=&{r6},=&{r7},=&{r8},{r0},{r3},{r4},{r5},~{cr0},~{ctr},~{memory},~{r11},~{r12}"(i64 342, i64 80871424, i64 undef, i64 0) #2, !srcloc !1
  br i1 undef, label %.lr.ph, label %.critedge15.preheader

.critedge15.preheader:                            ; preds = %_ZN10SubProcess12SafeSyscalls7recvmsgEiPNS0_13kernel_msghdrEi.exit.backedge, %.thread
  br i1 undef, label %.lr.ph93.preheader, label %.critedge15._crit_edge

.lr.ph93.preheader:                               ; preds = %.critedge15.preheader
  br label %.lr.ph93

.lr.ph:                                           ; preds = %_ZN10SubProcess12SafeSyscalls7recvmsgEiPNS0_13kernel_msghdrEi.exit.backedge, %.thread
  switch i32 undef, label %.critedge9 [
    i32 11, label %_ZN10SubProcess12SafeSyscalls7recvmsgEiPNS0_13kernel_msghdrEi.exit.backedge
    i32 4, label %_ZN10SubProcess12SafeSyscalls7recvmsgEiPNS0_13kernel_msghdrEi.exit.backedge
  ]

_ZN10SubProcess12SafeSyscalls7recvmsgEiPNS0_13kernel_msghdrEi.exit.backedge: ; preds = %.lr.ph, %.lr.ph
  br i1 undef, label %.lr.ph, label %.critedge15.preheader

.critedge9:                                       ; preds = %.lr.ph
  unreachable

.critedge15._crit_edge:                           ; preds = %.critedge15, %.critedge15.preheader
  br i1 undef, label %35, label %34

.lr.ph93:                                         ; preds = %.critedge15, %.lr.ph93.preheader
  switch i32 undef, label %33 [
    i32 0, label %23
    i32 1, label %23
    i32 2, label %23
    i32 3, label %23
    i32 4, label %23
    i32 5, label %23
    i32 6, label %23
    i32 7, label %23
    i32 8, label %27
    i32 9, label %30
  ]

; <label>:23                                      ; preds = %.lr.ph93, %.lr.ph93, %.lr.ph93, %.lr.ph93, %.lr.ph93, %.lr.ph93, %.lr.ph93, %.lr.ph93
  br i1 undef, label %24, label %.critedge15

; <label>:24                                      ; preds = %23
  br i1 undef, label %.critedge15, label %25

; <label>:25                                      ; preds = %24
  br i1 undef, label %.critedge15, label %26

; <label>:26                                      ; preds = %25
  unreachable

; <label>:27                                      ; preds = %.lr.ph93
  br i1 undef, label %.critedge15, label %28

; <label>:28                                      ; preds = %27
  br i1 undef, label %29, label %.critedge15

; <label>:29                                      ; preds = %28
  br label %.critedge15

; <label>:30                                      ; preds = %.lr.ph93
  br i1 undef, label %.critedge15, label %31

; <label>:31                                      ; preds = %30
  br i1 undef, label %32, label %.critedge15

; <label>:32                                      ; preds = %31
  br label %.critedge15

; <label>:33                                      ; preds = %.lr.ph93
  unreachable

.critedge15:                                      ; preds = %32, %31, %30, %29, %28, %27, %25, %24, %23
  br i1 undef, label %.lr.ph93, label %.critedge15._crit_edge

; <label>:34                                      ; preds = %.critedge15._crit_edge
  unreachable

; <label>:35                                      ; preds = %.critedge15._crit_edge
  br i1 undef, label %45, label %36

; <label>:36                                      ; preds = %35
  br i1 undef, label %37, label %38

; <label>:37                                      ; preds = %36
  br i1 undef, label %.preheader, label %38

.preheader:                                       ; preds = %37
  br i1 undef, label %.lr.ph101, label %._crit_edge

.lr.ph101:                                        ; preds = %.preheader
  br label %39

; <label>:38                                      ; preds = %37, %36
  unreachable

; <label>:39                                      ; preds = %43, %.lr.ph101
  br i1 undef, label %40, label %43

; <label>:40                                      ; preds = %39
  br i1 undef, label %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit17, label %41

; <label>:41                                      ; preds = %40
  unreachable

_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit17:   ; preds = %40
  br i1 undef, label %42, label %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit

; <label>:42                                      ; preds = %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit17
  unreachable

_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit:     ; preds = %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit17
  br i1 undef, label %.thread27, label %43

; <label>:43                                      ; preds = %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit, %39
  br i1 undef, label %39, label %._crit_edge

.thread27:                                        ; preds = %_ZN10SubProcess12SafeSyscalls5fcntlEiil.exit
  br label %58

._crit_edge:                                      ; preds = %43, %.preheader
  br i1 undef, label %.thread.outer, label %44

; <label>:44                                      ; preds = %._crit_edge
  unreachable

; <label>:45                                      ; preds = %35
  br i1 undef, label %46, label %.thread

; <label>:46                                      ; preds = %45
  br i1 undef, label %48, label %47

; <label>:47                                      ; preds = %46
  unreachable

; <label>:48                                      ; preds = %46
  br i1 undef, label %55, label %49

; <label>:49                                      ; preds = %48
  br i1 undef, label %50, label %51

; <label>:50                                      ; preds = %49
  br label %52

; <label>:51                                      ; preds = %49
  br label %52

; <label>:52                                      ; preds = %51, %50
  br label %53

; <label>:53                                      ; preds = %54, %52
  br i1 undef, label %54, label %.critedge13

; <label>:54                                      ; preds = %53
  br i1 undef, label %53, label %.critedge13

.critedge13:                                      ; preds = %54, %53
  br label %58

; <label>:55                                      ; preds = %48
  br label %56

; <label>:56                                      ; preds = %57, %55
  br i1 undef, label %57, label %.critedge14

; <label>:57                                      ; preds = %56
  br i1 undef, label %56, label %.critedge14

.critedge14:                                      ; preds = %57, %56
  br label %58

; <label>:58                                      ; preds = %.critedge14, %.critedge13, %.thread27
  br label %59

; <label>:59                                      ; preds = %58, %.critedge8
  br label %60

; <label>:60                                      ; preds = %59, %1
  ret void
}

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pwr8" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind argmemonly }
attributes #2 = { nounwind }

!1 = !{i32 -2140527538, i32 -2140527533}
