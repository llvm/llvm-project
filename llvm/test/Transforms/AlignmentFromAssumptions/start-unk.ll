; RUN: opt -passes=alignment-from-assumptions -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%type1 = type { %type2 }
%type2 = type { [4 x i8] }

; Function Attrs: nounwind
declare void @llvm.assume(i1) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.bswap.i32(i32) #1

; Function Attrs: nounwind uwtable
define void @test1() unnamed_addr #2 align 2 {

; CHECK-LABEL: @test1

entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  br i1 undef, label %return, label %if.end8

if.end8:                                          ; preds = %if.end
  br i1 undef, label %if.then13, label %if.end14

if.then13:                                        ; preds = %if.end8
  unreachable

if.end14:                                         ; preds = %if.end8
  br i1 undef, label %cond.false.i129, label %cond.end.i136

cond.false.i129:                                  ; preds = %if.end14
  unreachable

cond.end.i136:                                    ; preds = %if.end14
  br i1 undef, label %land.lhs.true.i, label %if.end.i145

land.lhs.true.i:                                  ; preds = %cond.end.i136
  br i1 undef, label %if.end.i145, label %if.then.i137

if.then.i137:                                     ; preds = %land.lhs.true.i
  br i1 undef, label %cond.false8.i, label %cond.end9.i

cond.false8.i:                                    ; preds = %if.then.i137
  unreachable

cond.end9.i:                                      ; preds = %if.then.i137
  br i1 undef, label %if.then23, label %if.end24

if.end.i145:                                      ; preds = %land.lhs.true.i, %cond.end.i136
  unreachable

if.then23:                                        ; preds = %cond.end9.i
  unreachable

if.end24:                                         ; preds = %cond.end9.i
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.end24
  unreachable

for.end:                                          ; preds = %if.end24
  br i1 undef, label %if.end123, label %if.then121

if.then121:                                       ; preds = %for.end
  unreachable

if.end123:                                        ; preds = %for.end
  br i1 undef, label %if.end150, label %if.then126

if.then126:                                       ; preds = %if.end123
  %ptrint.i.i185 = ptrtoint ptr undef to i64
  %maskedptr.i.i186 = and i64 %ptrint.i.i185, 1
  %maskcond.i.i187 = icmp eq i64 %maskedptr.i.i186, 0
  tail call void @llvm.assume(i1 %maskcond.i.i187) #0
  %ret.0.copyload.i.i189 = load i32, ptr undef, align 2

; CHECK: load {{.*}} align 2

  %0 = tail call i32 @llvm.bswap.i32(i32 %ret.0.copyload.i.i189) #0
  %conv131 = zext i32 %0 to i64
  %add.ptr132 = getelementptr inbounds i8, ptr undef, i64 %conv131
  br i1 undef, label %if.end150, label %if.end.i173

if.end.i173:                                      ; preds = %if.then126
  br i1 undef, label %test1.exit, label %cond.false.i.i.i.i174

cond.false.i.i.i.i174:                            ; preds = %if.end.i173
  unreachable

test1.exit: ; preds = %if.end.i173
  br i1 undef, label %test1a.exit, label %if.end.i124

if.end.i124:                                      ; preds = %test1.exit
  unreachable

test1a.exit: ; preds = %test1.exit
  br i1 undef, label %if.end150, label %for.body137.lr.ph

for.body137.lr.ph:                                ; preds = %test1a.exit
  br label %for.body137

for.body137:                                      ; preds = %test1b.exit, %for.body137.lr.ph
  %ShndxTable.0309 = phi ptr [ %add.ptr132, %for.body137.lr.ph ], [ %incdec.ptr, %test1b.exit ]
  br i1 undef, label %for.body137.if.end146_crit_edge, label %if.then140

for.body137.if.end146_crit_edge:                  ; preds = %for.body137
  %incdec.ptr = getelementptr inbounds %type1, ptr %ShndxTable.0309, i64 1
  br i1 undef, label %cond.false.i70, label %cond.end.i

if.then140:                                       ; preds = %for.body137
  %ret.0.copyload.i.i102 = load i32, ptr %ShndxTable.0309, align 2

; CHECK: load {{.*}} align 2

  unreachable

cond.false.i70:                                   ; preds = %for.body137.if.end146_crit_edge
  unreachable

cond.end.i:                                       ; preds = %for.body137.if.end146_crit_edge
  br i1 undef, label %test1b.exit, label %cond.false.i.i

cond.false.i.i:                                   ; preds = %cond.end.i
  unreachable

test1b.exit: ; preds = %cond.end.i
  br i1 undef, label %if.end150, label %for.body137

if.end150:                                        ; preds = %test1b.exit, %test1a.exit, %if.then126, %if.end123
  br i1 undef, label %for.end176, label %for.body155.lr.ph

for.body155.lr.ph:                                ; preds = %if.end150
  unreachable

for.end176:                                       ; preds = %if.end150
  unreachable

return:                                           ; preds = %if.end
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind uwtable }

