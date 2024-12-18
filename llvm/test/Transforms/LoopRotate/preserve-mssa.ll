; RUN: opt -S -passes=loop-rotate -verify-memoryssa < %s | FileCheck %s

; CHECK-LABEL: @multiedge(
define void @multiedge(i1 %arg, i32 %arg2) {
entry:
  br label %retry

retry:                                            ; preds = %sw.epilog, %entry
  br i1 %arg, label %cleanup, label %if.end

if.end:                                           ; preds = %retry
  switch i32 %arg2, label %sw.epilog [
    i32 -3, label %cleanup
    i32 -5, label %cleanup
    i32 -16, label %cleanup
    i32 -25, label %cleanup
  ]

sw.epilog:                                        ; preds = %if.end
  br label %retry

cleanup:                                          ; preds = %if.end, %if.end, %if.end, %if.end, %retry
  ret void
}

; CHECK-LABEL: @read_line(
define internal fastcc i32 @read_line(ptr nocapture %f, i1 %arg) unnamed_addr {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %call = call ptr @prepbuffer(ptr nonnull undef)
  %call1 = call ptr @fgets(ptr %call, i32 8192, ptr %f)
  br i1 %arg, label %if.then, label %if.end

if.then:                                          ; preds = %for.cond
  ret i32 undef

if.end:                                           ; preds = %for.cond
  %call4 = call i64 @strlen(ptr %call)
  br label %for.cond
}

declare dso_local ptr @prepbuffer(ptr) local_unnamed_addr
declare dso_local ptr @fgets(ptr, i32, ptr nocapture) local_unnamed_addr
declare dso_local i64 @strlen(ptr nocapture) local_unnamed_addr


; CHECK-LABEL: @loop3
define dso_local fastcc void @loop3(i1 %arg) unnamed_addr {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  br i1 %arg, label %for.body, label %for.end81

for.body:                                         ; preds = %for.cond
  %.idx122.val = load i32, ptr undef, align 8
  call fastcc void @cont()
  br label %for.cond

for.end81:                                        ; preds = %for.cond
  ret void
}

; CHECK-LABEL: @loop4
define dso_local fastcc void @loop4(i1 %arg) unnamed_addr {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  br i1 %arg, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  call fastcc void @cont()
  br label %while.cond

while.end:                                        ; preds = %while.cond
  call fastcc void @cont()
  call fastcc void @cont()
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
declare dso_local fastcc void @cont() unnamed_addr

@glob_array = internal unnamed_addr constant [3 x i32] [i32 1, i32 0, i32 2], align 4
; Test against failure in MemorySSAUpdater, when rotate clones instructions as Value.
; CHECK-LABEL: @loop5
define dso_local fastcc void @loop5(i1 %arg) unnamed_addr {
entry:
  br label %for.body

do.cond:                          ; preds = %for.body
  unreachable

for.body:                               ; preds = %if.end, %entry
  %indvar = phi i64 [ %indvar.next, %if.end ], [ 0, %entry ]
  %array = getelementptr inbounds [3 x i32], ptr @glob_array, i64 0, i64 %indvar
  %0 = load i32, ptr %array, align 4
  br i1 %arg, label %do.cond, label %if.end

if.end:                                 ; preds = %for.body
  store i32 undef, ptr undef, align 4
  %indvar.next = add nuw nsw i64 %indvar, 1
  br label %for.body
}


