; RUN: llc -regalloc=greedy -verify-coalescing -mtriple=x86_64-unknown-linux-gnu < %s
; Check the live range is updated properly after register coalescing.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.str = internal unnamed_addr constant { [17 x i8], [47 x i8] } { [17 x i8] c"0123456789ABCDEF\00", [47 x i8] zeroinitializer }, align 32
@b = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i32* null, align 8
@__sancov_gen_cov = private global [9 x i32] zeroinitializer

; Function Attrs: nounwind sanitize_address
define void @fn2(i8* %p1) local_unnamed_addr #0 {
entry:
  %0 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 4) to i32*) monotonic, align 4
  %1 = icmp sge i32 0, %0
  br i1 %1, label %2, label %3

; <label>:2:                                      ; preds = %entry
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 4) to i32*))
  call void asm sideeffect "", ""()
  br label %3

; <label>:3:                                      ; preds = %entry, %2
  br label %while.cond.outer

while.cond.outer:                                 ; preds = %75, %3
  %e.0.ph = phi i8* [ %e.058, %75 ], [ undef, %3 ]
  %c.0.ph = phi i32* [ %c.059, %75 ], [ undef, %3 ]
  %p1.addr.0.ph = phi i8* [ %incdec.ptr60, %75 ], [ %p1, %3 ]
  %4 = ptrtoint i8* %p1.addr.0.ph to i64
  %5 = lshr i64 %4, 3
  %6 = add i64 %5, 2147450880
  %7 = inttoptr i64 %6 to i8*
  %8 = load i8, i8* %7
  %9 = icmp ne i8 %8, 0
  br i1 %9, label %10, label %15

; <label>:10:                                     ; preds = %while.cond.outer
  %11 = and i64 %4, 7
  %12 = trunc i64 %11 to i8
  %13 = icmp sge i8 %12, %8
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %10
  call void @__asan_report_load1(i64 %4)
  call void asm sideeffect "", ""()
  unreachable

; <label>:15:                                     ; preds = %10, %while.cond.outer
  %16 = load i8, i8* %p1.addr.0.ph, align 1
  call void @__sanitizer_cov_trace_cmp1(i8 %16, i8 0)
  %cmp57 = icmp eq i8 %16, 0
  br i1 %cmp57, label %while.cond.outer.enoent.loopexit96_crit_edge, label %while.body.preheader

while.cond.outer.enoent.loopexit96_crit_edge:     ; preds = %15
  %17 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 8) to i32*) monotonic, align 4
  %18 = icmp sge i32 0, %17
  br i1 %18, label %19, label %20

; <label>:19:                                     ; preds = %while.cond.outer.enoent.loopexit96_crit_edge
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 8) to i32*))
  call void asm sideeffect "", ""()
  br label %20

; <label>:20:                                     ; preds = %while.cond.outer.enoent.loopexit96_crit_edge, %19
  br label %enoent.loopexit96

while.body.preheader:                             ; preds = %15
  br label %while.body

while.body:                                       ; preds = %56, %while.body.preheader
  %21 = phi i8 [ %52, %56 ], [ %16, %while.body.preheader ]
  %p1.addr.0.ph.pn = phi i8* [ %incdec.ptr60, %56 ], [ %p1.addr.0.ph, %while.body.preheader ]
  %c.059 = phi i32* [ %incdec.ptr18, %56 ], [ %c.0.ph, %while.body.preheader ]
  %e.058 = phi i8* [ %incdec.ptr60, %56 ], [ %e.0.ph, %while.body.preheader ]
  %incdec.ptr60 = getelementptr inbounds i8, i8* %p1.addr.0.ph.pn, i64 1
  %conv = sext i8 %21 to i32
  %call = tail call i32 (i8*, i32, ...) bitcast (i32 (...)* @fn3 to i32 (i8*, i32, ...)*)(i8* getelementptr inbounds ({ [17 x i8], [47 x i8] }, { [17 x i8], [47 x i8] }* @.str, i32 0, i32 0, i64 0), i32 %conv) #2
  call void @__sanitizer_cov_trace_cmp4(i32 %call, i32 0)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.end5, label %cleanup

if.end5:                                          ; preds = %while.body
  call void @__sanitizer_cov_trace_cmp1(i8 %21, i8 58)
  %cmp6 = icmp eq i8 %21, 58
  br i1 %cmp6, label %if.end14, label %cleanup.thread40

if.end14:                                         ; preds = %if.end5
  %22 = load i8, i8* inttoptr (i64 add (i64 lshr (i64 ptrtoint (i32** @a to i64), i64 3), i64 2147450880) to i8*)
  %23 = icmp ne i8 %22, 0
  br i1 %23, label %24, label %25

; <label>:24:                                     ; preds = %if.end14
  call void @__asan_report_load8(i64 ptrtoint (i32** @a to i64))
  call void asm sideeffect "", ""()
  unreachable

; <label>:25:                                     ; preds = %if.end14
  %26 = load i32*, i32** @a, align 8
  %tobool15 = icmp eq i32* %26, null
  br i1 %tobool15, label %cleanup.thread39, label %cleanup23.loopexit

cleanup.thread39:                                 ; preds = %25
  %incdec.ptr18 = getelementptr inbounds i32, i32* %c.059, i64 1
  %27 = ptrtoint i32* %c.059 to i64
  %28 = lshr i64 %27, 3
  %29 = add i64 %28, 2147450880
  %30 = inttoptr i64 %29 to i8*
  %31 = load i8, i8* %30
  %32 = icmp ne i8 %31, 0
  br i1 %32, label %33, label %39

; <label>:33:                                     ; preds = %cleanup.thread39
  %34 = and i64 %27, 7
  %35 = add i64 %34, 3
  %36 = trunc i64 %35 to i8
  %37 = icmp sge i8 %36, %31
  br i1 %37, label %38, label %39

; <label>:38:                                     ; preds = %33
  call void @__asan_report_store4(i64 %27)
  call void asm sideeffect "", ""()
  unreachable

; <label>:39:                                     ; preds = %33, %cleanup.thread39
  store i32 0, i32* %c.059, align 4
  %40 = ptrtoint i8* %incdec.ptr60 to i64
  %41 = lshr i64 %40, 3
  %42 = add i64 %41, 2147450880
  %43 = inttoptr i64 %42 to i8*
  %44 = load i8, i8* %43
  %45 = icmp ne i8 %44, 0
  br i1 %45, label %46, label %51

; <label>:46:                                     ; preds = %39
  %47 = and i64 %40, 7
  %48 = trunc i64 %47 to i8
  %49 = icmp sge i8 %48, %44
  br i1 %49, label %50, label %51

; <label>:50:                                     ; preds = %46
  call void @__asan_report_load1(i64 %40)
  call void asm sideeffect "", ""()
  unreachable

; <label>:51:                                     ; preds = %46, %39
  %52 = load i8, i8* %incdec.ptr60, align 1
  call void @__sanitizer_cov_trace_cmp1(i8 %52, i8 0)
  %cmp = icmp eq i8 %52, 0
  br i1 %cmp, label %enoent.loopexit, label %cleanup.thread39.while.body_crit_edge

cleanup.thread39.while.body_crit_edge:            ; preds = %51
  %53 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 12) to i32*) monotonic, align 4
  %54 = icmp sge i32 0, %53
  br i1 %54, label %55, label %56

; <label>:55:                                     ; preds = %cleanup.thread39.while.body_crit_edge
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 12) to i32*))
  call void asm sideeffect "", ""()
  br label %56

; <label>:56:                                     ; preds = %cleanup.thread39.while.body_crit_edge, %55
  br label %while.body

cleanup.thread40:                                 ; preds = %if.end5
  %57 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 16) to i32*) monotonic, align 4
  %58 = icmp sge i32 0, %57
  br i1 %58, label %59, label %60

; <label>:59:                                     ; preds = %cleanup.thread40
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 16) to i32*))
  call void asm sideeffect "", ""()
  br label %60

; <label>:60:                                     ; preds = %cleanup.thread40, %59
  %call20 = tail call i32 (i8*, ...) bitcast (i32 (...)* @fn4 to i32 (i8*, ...)*)(i8* %e.058) #2
  br label %enoent

cleanup:                                          ; preds = %while.body
  %61 = load i8, i8* inttoptr (i64 add (i64 lshr (i64 ptrtoint (i32* @b to i64), i64 3), i64 2147450880) to i8*)
  %62 = icmp ne i8 %61, 0
  br i1 %62, label %63, label %66

; <label>:63:                                     ; preds = %cleanup
  %64 = icmp sge i8 trunc (i64 add (i64 and (i64 ptrtoint (i32* @b to i64), i64 7), i64 3) to i8), %61
  br i1 %64, label %65, label %66

; <label>:65:                                     ; preds = %63
  call void @__asan_report_load4(i64 ptrtoint (i32* @b to i64))
  call void asm sideeffect "", ""()
  unreachable

; <label>:66:                                     ; preds = %63, %cleanup
  %67 = load i32, i32* @b, align 4
  call void @__sanitizer_cov_trace_cmp4(i32 %67, i32 0)
  %tobool3 = icmp eq i32 %67, 0
  br i1 %tobool3, label %cleanup.while.cond.outer_crit_edge, label %cleanup.enoent.loopexit96_crit_edge

cleanup.enoent.loopexit96_crit_edge:              ; preds = %66
  %68 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 20) to i32*) monotonic, align 4
  %69 = icmp sge i32 0, %68
  br i1 %69, label %70, label %71

; <label>:70:                                     ; preds = %cleanup.enoent.loopexit96_crit_edge
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 20) to i32*))
  call void asm sideeffect "", ""()
  br label %71

; <label>:71:                                     ; preds = %cleanup.enoent.loopexit96_crit_edge, %70
  br label %enoent.loopexit96

cleanup.while.cond.outer_crit_edge:               ; preds = %66
  %72 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 24) to i32*) monotonic, align 4
  %73 = icmp sge i32 0, %72
  br i1 %73, label %74, label %75

; <label>:74:                                     ; preds = %cleanup.while.cond.outer_crit_edge
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 24) to i32*))
  call void asm sideeffect "", ""()
  br label %75

; <label>:75:                                     ; preds = %cleanup.while.cond.outer_crit_edge, %74
  br label %while.cond.outer

enoent.loopexit:                                  ; preds = %51
  %76 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 28) to i32*) monotonic, align 4
  %77 = icmp sge i32 0, %76
  br i1 %77, label %78, label %79

; <label>:78:                                     ; preds = %enoent.loopexit
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 28) to i32*))
  call void asm sideeffect "", ""()
  br label %79

; <label>:79:                                     ; preds = %enoent.loopexit, %78
  br label %enoent

enoent.loopexit96:                                ; preds = %71, %20
  br label %enoent

enoent:                                           ; preds = %enoent.loopexit96, %79, %60
  %call22 = tail call i32* (...) @fn1() #2
  br label %cleanup23

cleanup23.loopexit:                               ; preds = %25
  %80 = load atomic i32, i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 32) to i32*) monotonic, align 4
  %81 = icmp sge i32 0, %80
  br i1 %81, label %82, label %83

; <label>:82:                                     ; preds = %cleanup23.loopexit
  call void @__sanitizer_cov(i32* inttoptr (i64 add (i64 ptrtoint ([9 x i32]* @__sancov_gen_cov to i64), i64 32) to i32*))
  call void asm sideeffect "", ""()
  br label %83

; <label>:83:                                     ; preds = %cleanup23.loopexit, %82
  br label %cleanup23

cleanup23:                                        ; preds = %83, %enoent
  ret void
}

declare i32 @fn3(...) local_unnamed_addr #1

declare i32 @fn4(...) local_unnamed_addr #1

declare i32* @fn1(...) local_unnamed_addr #1

declare void @__sanitizer_cov(i32*)

declare void @__sanitizer_cov_trace_cmp1(i8, i8)

declare void @__sanitizer_cov_trace_cmp4(i32, i32)

declare void @__asan_report_load1(i64)

declare void @__asan_report_load4(i64)

declare void @__asan_report_load8(i64)

declare void @__asan_report_store4(i64)

