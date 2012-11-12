; RUN: llc < %s -mtriple=armv7-apple-darwin -O3 -mcpu=arm1136jf-s
; PR7421

%struct.CONTENTBOX = type { i32, i32, i32, i32, i32 }
%struct.FILE = type { i8* }
%struct.tilebox = type { %struct.tilebox*, double, double, double, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.UNCOMBOX = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.cellbox = type { i8*, i32, i32, i32, [9 x i32], i32, i32, i32, i32, i32, i32, i32, double, double, double, double, double, i32, i32, %struct.CONTENTBOX*, %struct.UNCOMBOX*, [8 x %struct.tilebox*] }
%struct.termbox = type { %struct.termbox*, i32, i32, i32, i32, i32 }

@.str2708 = external constant [14 x i8], align 4  ; <[14 x i8]*> [#uses=1]

define void @TW_oldinput(%struct.FILE* nocapture %fp) nounwind {
entry:
  %xcenter = alloca i32, align 4                  ; <i32*> [#uses=2]
  %0 = call i32 (%struct.FILE*, i8*, ...)* @fscanf(%struct.FILE* %fp, i8* getelementptr inbounds ([14 x i8]* @.str2708, i32 0, i32 0), i32* undef, i32* undef, i32* %xcenter, i32* null) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 4                          ; <i1> [#uses=1]
  br i1 %1, label %bb, label %return

bb:                                               ; preds = %bb445, %entry
  %2 = load %struct.cellbox** undef, align 4      ; <%struct.cellbox*> [#uses=2]
  %3 = getelementptr inbounds %struct.cellbox* %2, i32 0, i32 3 ; <i32*> [#uses=1]
  store i32 undef, i32* %3, align 4
  %4 = load i32* undef, align 4                   ; <i32> [#uses=3]
  %5 = icmp eq i32 undef, 1                       ; <i1> [#uses=1]
  br i1 %5, label %bb10, label %bb445

bb10:                                             ; preds = %bb
  br i1 undef, label %bb11, label %bb445

bb11:                                             ; preds = %bb10
  %6 = load %struct.tilebox** undef, align 4      ; <%struct.tilebox*> [#uses=3]
  %7 = load %struct.termbox** null, align 4       ; <%struct.termbox*> [#uses=1]
  %8 = getelementptr inbounds %struct.tilebox* %6, i32 0, i32 13 ; <i32*> [#uses=1]
  %9 = load i32* %8, align 4                      ; <i32> [#uses=3]
  %10 = getelementptr inbounds %struct.tilebox* %6, i32 0, i32 15 ; <i32*> [#uses=1]
  %11 = load i32* %10, align 4                    ; <i32> [#uses=1]
  br i1 false, label %bb12, label %bb13

bb12:                                             ; preds = %bb11
  unreachable

bb13:                                             ; preds = %bb11
  %iftmp.40.0.neg = sdiv i32 0, -2                ; <i32> [#uses=2]
  %12 = sub nsw i32 0, %9                         ; <i32> [#uses=1]
  %13 = sitofp i32 %12 to double                  ; <double> [#uses=1]
  %14 = fdiv double %13, 0.000000e+00             ; <double> [#uses=1]
  %15 = fptosi double %14 to i32                  ; <i32> [#uses=1]
  %iftmp.41.0.in = add i32 0, %15                 ; <i32> [#uses=1]
  %iftmp.41.0.neg = sdiv i32 %iftmp.41.0.in, -2   ; <i32> [#uses=3]
  br i1 undef, label %bb43.loopexit, label %bb21

bb21:                                             ; preds = %bb13
  %16 = fptosi double undef to i32                ; <i32> [#uses=1]
  %17 = fsub double undef, 0.000000e+00           ; <double> [#uses=1]
  %not.460 = fcmp oge double %17, 5.000000e-01    ; <i1> [#uses=1]
  %18 = zext i1 %not.460 to i32                   ; <i32> [#uses=1]
  %iftmp.42.0 = add i32 %16, %iftmp.41.0.neg      ; <i32> [#uses=1]
  %19 = add i32 %iftmp.42.0, %18                  ; <i32> [#uses=1]
  store i32 %19, i32* undef, align 4
  %20 = sub nsw i32 0, %9                         ; <i32> [#uses=1]
  %21 = sitofp i32 %20 to double                  ; <double> [#uses=1]
  %22 = fdiv double %21, 0.000000e+00             ; <double> [#uses=2]
  %23 = fptosi double %22 to i32                  ; <i32> [#uses=1]
  %24 = fsub double %22, undef                    ; <double> [#uses=1]
  %not.461 = fcmp oge double %24, 5.000000e-01    ; <i1> [#uses=1]
  %25 = zext i1 %not.461 to i32                   ; <i32> [#uses=1]
  %iftmp.43.0 = add i32 %23, %iftmp.41.0.neg      ; <i32> [#uses=1]
  %26 = add i32 %iftmp.43.0, %25                  ; <i32> [#uses=1]
  %27 = getelementptr inbounds %struct.tilebox* %6, i32 0, i32 10 ; <i32*> [#uses=1]
  store i32 %26, i32* %27, align 4
  %28 = fptosi double undef to i32                ; <i32> [#uses=1]
  %iftmp.45.0 = add i32 %28, %iftmp.40.0.neg      ; <i32> [#uses=1]
  %29 = add i32 %iftmp.45.0, 0                    ; <i32> [#uses=1]
  store i32 %29, i32* undef, align 4
  br label %bb43.loopexit

bb36:                                             ; preds = %bb43.loopexit, %bb36
  %termptr.0478 = phi %struct.termbox* [ %42, %bb36 ], [ %7, %bb43.loopexit ] ; <%struct.termbox*> [#uses=1]
  %30 = load i32* undef, align 4                  ; <i32> [#uses=1]
  %31 = sub nsw i32 %30, %9                       ; <i32> [#uses=1]
  %32 = sitofp i32 %31 to double                  ; <double> [#uses=1]
  %33 = fdiv double %32, 0.000000e+00             ; <double> [#uses=1]
  %34 = fptosi double %33 to i32                  ; <i32> [#uses=1]
  %iftmp.46.0 = add i32 %34, %iftmp.41.0.neg      ; <i32> [#uses=1]
  %35 = add i32 %iftmp.46.0, 0                    ; <i32> [#uses=1]
  store i32 %35, i32* undef, align 4
  %36 = sub nsw i32 0, %11                        ; <i32> [#uses=1]
  %37 = sitofp i32 %36 to double                  ; <double> [#uses=1]
  %38 = fmul double %37, 0.000000e+00             ; <double> [#uses=1]
  %39 = fptosi double %38 to i32                  ; <i32> [#uses=1]
  %iftmp.47.0 = add i32 %39, %iftmp.40.0.neg      ; <i32> [#uses=1]
  %40 = add i32 %iftmp.47.0, 0                    ; <i32> [#uses=1]
  store i32 %40, i32* undef, align 4
  %41 = getelementptr inbounds %struct.termbox* %termptr.0478, i32 0, i32 0 ; <%struct.termbox**> [#uses=1]
  %42 = load %struct.termbox** %41, align 4       ; <%struct.termbox*> [#uses=2]
  %43 = icmp eq %struct.termbox* %42, null        ; <i1> [#uses=1]
  br i1 %43, label %bb52.loopexit, label %bb36

bb43.loopexit:                                    ; preds = %bb21, %bb13
  br i1 undef, label %bb52.loopexit, label %bb36

bb52.loopexit:                                    ; preds = %bb43.loopexit, %bb36
  %44 = icmp eq i32 %4, 0                         ; <i1> [#uses=1]
  br i1 %44, label %bb.nph485, label %bb54

bb54:                                             ; preds = %bb52.loopexit
  switch i32 %4, label %bb62 [
    i32 2, label %bb56
    i32 3, label %bb57
  ]

bb56:                                             ; preds = %bb54
  br label %bb62

bb57:                                             ; preds = %bb54
  br label %bb62

bb62:                                             ; preds = %bb57, %bb56, %bb54
  unreachable

bb.nph485:                                        ; preds = %bb52.loopexit
  br label %bb248

bb248:                                            ; preds = %bb322, %bb.nph485
  %45 = icmp eq i32 undef, %4                     ; <i1> [#uses=1]
  br i1 %45, label %bb322, label %bb249

bb249:                                            ; preds = %bb248
  %46 = getelementptr inbounds %struct.cellbox* %2, i32 0, i32 21, i32 undef ; <%struct.tilebox**> [#uses=1]
  %47 = load %struct.tilebox** %46, align 4       ; <%struct.tilebox*> [#uses=1]
  %48 = getelementptr inbounds %struct.tilebox* %47, i32 0, i32 11 ; <i32*> [#uses=1]
  store i32 undef, i32* %48, align 4
  unreachable

bb322:                                            ; preds = %bb248
  br i1 undef, label %bb248, label %bb445

bb445:                                            ; preds = %bb322, %bb10, %bb
  %49 = call i32 (%struct.FILE*, i8*, ...)* @fscanf(%struct.FILE* %fp, i8* getelementptr inbounds ([14 x i8]* @.str2708, i32 0, i32 0), i32* undef, i32* undef, i32* %xcenter, i32* null) nounwind ; <i32> [#uses=1]
  %50 = icmp eq i32 %49, 4                        ; <i1> [#uses=1]
  br i1 %50, label %bb, label %return

return:                                           ; preds = %bb445, %entry
  ret void
}

declare i32 @fscanf(%struct.FILE* nocapture, i8* nocapture, ...) nounwind
