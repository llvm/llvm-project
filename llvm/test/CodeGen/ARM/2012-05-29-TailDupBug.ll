; RUN: llc -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -verify-machineinstrs < %s

; Teach taildup to update livein set to appease verifier.
; rdar://11538365

%struct.__CFString.2 = type opaque

declare void @CFRelease(ptr)

define hidden fastcc i32 @t() ssp {
entry:
  %mylocale.i.i = alloca [256 x i8], align 1
  br i1 undef, label %return, label %CFStringIsHyphenationAvailableForLocale.exit

CFStringIsHyphenationAvailableForLocale.exit:     ; preds = %entry
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %CFStringIsHyphenationAvailableForLocale.exit
  br i1 undef, label %if.end8.thread.i, label %if.then.i

if.then.i:                                        ; preds = %if.end
  br i1 undef, label %if.end8.thread.i, label %if.end8.i

if.end8.thread.i:                                 ; preds = %if.then.i, %if.end
  unreachable

if.end8.i:                                        ; preds = %if.then.i
  br i1 undef, label %if.then11.i, label %__CFHyphenationPullTokenizer.exit

if.then11.i:                                      ; preds = %if.end8.i
  unreachable

__CFHyphenationPullTokenizer.exit:                ; preds = %if.end8.i
  br i1 undef, label %if.end68, label %if.then3

if.then3:                                         ; preds = %__CFHyphenationPullTokenizer.exit
  br i1 undef, label %cond.end, label %cond.false

cond.false:                                       ; preds = %if.then3
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %if.then3
  br i1 undef, label %while.end, label %while.body

while.body:                                       ; preds = %cond.end
  unreachable

while.end:                                        ; preds = %cond.end
  br i1 undef, label %if.end5.i, label %if.then.i16

if.then.i16:                                      ; preds = %while.end
  br i1 undef, label %if.then4.i, label %if.end5.i

if.then4.i:                                       ; preds = %if.then.i16
  br i1 false, label %cleanup.thread, label %if.end.i20

if.end5.i:                                        ; preds = %if.then.i16, %while.end
  unreachable

if.end.i20:                                       ; preds = %if.then4.i
  br label %for.body.i146.i

for.body.i146.i:                                  ; preds = %for.body.i146.i, %if.end.i20
  br i1 undef, label %if.end20.i, label %for.body.i146.i

if.end20.i:                                       ; preds = %for.body.i146.i
  br i1 undef, label %cleanup.thread, label %if.end23.i

if.end23.i:                                       ; preds = %if.end20.i
  br label %for.body.i94.i

for.body.i94.i:                                   ; preds = %for.body.i94.i, %if.end23.i
  br i1 undef, label %if.then28.i, label %for.body.i94.i

if.then28.i:                                      ; preds = %for.body.i94.i
  br i1 undef, label %cond.true.i26, label %land.lhs.true

cond.true.i26:                                    ; preds = %if.then28.i
  br label %land.lhs.true

land.lhs.true:                                    ; preds = %cond.true.i26, %if.then28.i
  br i1 false, label %cleanup.thread, label %if.end35

if.end35:                                         ; preds = %land.lhs.true
  br i1 undef, label %cleanup.thread, label %if.end45

if.end45:                                         ; preds = %if.end35
  br i1 undef, label %if.then50, label %if.end.i37

if.end.i37:                                       ; preds = %if.end45
  br label %if.then50

if.then50:                                        ; preds = %if.end.i37, %if.end45
  br i1 undef, label %__CFHyphenationGetHyphensForString.exit, label %if.end.i

if.end.i:                                         ; preds = %if.then50
  br i1 undef, label %cleanup.i, label %cond.true.i

cond.true.i:                                      ; preds = %if.end.i
  br i1 undef, label %for.cond16.preheader.i, label %for.cond57.preheader.i

for.cond16.preheader.i:                           ; preds = %cond.true.i
  %cmp1791.i = icmp sgt i32 undef, 1
  br i1 %cmp1791.i, label %for.body18.i, label %for.cond57.preheader.i

for.cond57.preheader.i:                           ; preds = %for.cond16.preheader.i, %cond.true.i
  %sub69.i = add i32 undef, -2
  br label %cleanup.i

for.body18.i:                                     ; preds = %for.cond16.preheader.i
  store i16 0, ptr undef, align 2
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %for.body18.i
  br label %while.body.i

cleanup.i:                                        ; preds = %for.cond57.preheader.i, %if.end.i
  br label %__CFHyphenationGetHyphensForString.exit

__CFHyphenationGetHyphensForString.exit:          ; preds = %cleanup.i, %if.then50
  %retval.1.i = phi i32 [ 0, %cleanup.i ], [ -1, %if.then50 ]
  br label %if.end68

cleanup.thread:                                   ; preds = %if.end35, %land.lhs.true, %if.end20.i, %if.then4.i
  call void @llvm.stackrestore(ptr null)
  br label %return

if.end68:                                         ; preds = %__CFHyphenationGetHyphensForString.exit, %__CFHyphenationPullTokenizer.exit
  %hyphenCount.2 = phi i32 [ %retval.1.i, %__CFHyphenationGetHyphensForString.exit ], [ 0, %__CFHyphenationPullTokenizer.exit ]
  %_token.1 = phi ptr [ null, %__CFHyphenationGetHyphensForString.exit ], [ undef, %__CFHyphenationPullTokenizer.exit ]
  call void @CFRelease(ptr %_token.1)
  br label %return

return:                                           ; preds = %if.end68, %cleanup.thread, %CFStringIsHyphenationAvailableForLocale.exit, %entry
  %retval.1 = phi i32 [ %hyphenCount.2, %if.end68 ], [ -1, %CFStringIsHyphenationAvailableForLocale.exit ], [ -1, %cleanup.thread ], [ -1, %entry ]
  ret i32 %retval.1
}

declare void @llvm.stackrestore(ptr) nounwind
