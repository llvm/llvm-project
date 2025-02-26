; RUN: llc < %s -mtriple=arm64-apple-ios
; rdar://11849816

@shlib_path_substitutions = external hidden unnamed_addr global ptr, align 8

declare i64 @llvm.objectsize.i64(ptr, i1) nounwind readnone

declare noalias ptr @xmalloc(i64) optsize

declare i64 @strlen(ptr nocapture) nounwind readonly optsize

declare ptr @__strcpy_chk(ptr, ptr, i64) nounwind optsize

declare ptr @__strcat_chk(ptr, ptr, i64) nounwind optsize

declare noalias ptr @xstrdup(ptr) optsize

define ptr @dyld_fix_path(ptr %path, i1 %arg) nounwind optsize ssp {
entry:
  br i1  %arg, label %if.end56, label %for.cond

for.cond:                                         ; preds = %entry
  br i1  %arg, label %for.cond10, label %for.body

for.body:                                         ; preds = %for.cond
  unreachable

for.cond10:                                       ; preds = %for.cond
  br i1  %arg, label %if.end56, label %for.body14

for.body14:                                       ; preds = %for.cond10
  %call22 = tail call i64 @strlen(ptr undef) nounwind optsize
  %sext = shl i64 %call22, 32
  %conv30 = ashr exact i64 %sext, 32
  %add29 = sub i64 0, %conv30
  %sub = add i64 %add29, 0
  %add31 = shl i64 %sub, 32
  %sext59 = add i64 %add31, 4294967296
  %conv33 = ashr exact i64 %sext59, 32
  %call34 = tail call noalias ptr @xmalloc(i64 %conv33) nounwind optsize
  br i1  %arg, label %cond.false45, label %cond.true43

cond.true43:                                      ; preds = %for.body14
  unreachable

cond.false45:                                     ; preds = %for.body14
  %add.ptr = getelementptr inbounds i8, ptr %path, i64 %conv30
  unreachable

if.end56:                                         ; preds = %for.cond10, %entry
  ret ptr null
}

declare i32 @strncmp(ptr nocapture, ptr nocapture, i64) nounwind readonly optsize

declare ptr @strcpy(ptr, ptr nocapture) nounwind
