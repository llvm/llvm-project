; RUN: llc < %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-elf"

%struct.httpd_fs_file = type { ptr, i16 }
%struct.psock = type { %struct.pt, %struct.pt, ptr, ptr, ptr, i16, i16, %struct.httpd_fs_file, i16, i8, i8 }
%struct.pt = type { i16 }

@foo = external global ptr

define signext i8 @psock_readto(ptr nocapture %psock, i8 zeroext %c) nounwind {
entry:
  switch i16 undef, label %sw.epilog [
    i16 0, label %sw.bb
    i16 283, label %if.else.i
  ]

sw.bb:                                            ; preds = %entry
  br label %do.body

do.body:                                          ; preds = %while.cond36.i, %while.end.i, %sw.bb
  br label %while.cond.i

if.else.i:                                        ; preds = %entry
  br i1 undef, label %psock_newdata.exit, label %if.else11.i

if.else11.i:                                      ; preds = %if.else.i
  ret i8 0

psock_newdata.exit:                               ; preds = %if.else.i
  ret i8 0

while.cond.i:                                     ; preds = %while.body.i, %do.body
  br i1 undef, label %while.end.i, label %while.body.i

while.body.i:                                     ; preds = %while.cond.i
  br i1 undef, label %do.end41, label %while.cond.i

while.end.i:                                      ; preds = %while.cond.i
  br i1 undef, label %do.body, label %while.cond36.i.preheader

while.cond36.i.preheader:                         ; preds = %while.end.i
  br label %while.cond36.i

while.cond36.i:                                   ; preds = %while.body41.i, %while.cond36.i.preheader
  br i1 undef, label %do.body, label %while.body41.i

while.body41.i:                                   ; preds = %while.cond36.i
  %tmp43.i = load ptr, ptr @foo                      ; <i8*> [#uses=2]
  %tmp44.i = load i8, ptr %tmp43.i                    ; <i8> [#uses=1]
  %ptrincdec50.i = getelementptr inbounds i8, ptr %tmp43.i, i16 1 ; <i8*> [#uses=1]
  store ptr %ptrincdec50.i, ptr @foo
  %cmp55.i = icmp eq i8 %tmp44.i, %c              ; <i1> [#uses=1]
  br i1 %cmp55.i, label %do.end41, label %while.cond36.i

do.end41:                                         ; preds = %while.body41.i, %while.body.i
  br i1 undef, label %if.then46, label %sw.epilog

if.then46:                                        ; preds = %do.end41
  ret i8 0

sw.epilog:                                        ; preds = %do.end41, %entry
  ret i8 2
}
