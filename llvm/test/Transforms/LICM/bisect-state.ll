; Make sure we don't crash in LICM.
; RUN: opt %s -passes=licm -opt-bisect-limit=1

define void @patatino() {
for.cond1:
  br label %for.body
for.body:
  br label %for.cond5
for.cond5:
  br i1 true, label %if.end, label %for.end
if.end:
  br label %for.cond5
for.end:
  br label %for.body
}
