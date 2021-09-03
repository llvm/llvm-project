; RUN: opt < %s -passes='print<block-freq>' -disable-output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @fn1(i32* %f) {
entry:
  %tobool7 = icmp eq i32 undef, 0
  br i1 undef, label %if.end.12, label %for.body.5

for.inc:
  store i32 undef, i32* %f, align 4
  br label %for.body.5

for.body.5:                                       ; preds = %for.cond.4.preheader
  br i1 %tobool7, label %for.inc.9, label %for.inc

for.inc.9:                                        ; preds = %for.body.5
  br i1 %tobool7, label %for.inc.9.1, label %for.inc

if.end.12:                                        ; preds = %if.end.12, %for.body
  br i1 undef, label %for.end.17, label %for.inc

for.end.17:                                       ; preds = %entry
  ret void

for.inc.9.1:                                      ; preds = %for.inc.9
  br i1 %tobool7, label %for.inc.9.2, label %for.inc

for.inc.9.2:                                      ; preds = %for.inc.9.1
  br i1 %tobool7, label %for.inc.9.3, label %for.inc

for.inc.9.3:                                      ; preds = %for.inc.9.2
  br i1 %tobool7, label %for.inc.9.4, label %for.inc

for.inc.9.4:                                      ; preds = %for.inc.9.3
  br i1 %tobool7, label %for.inc.9.5, label %for.inc

for.inc.9.5:                                      ; preds = %for.inc.9.4
  br i1 %tobool7, label %for.inc.9.6, label %for.inc

for.inc.9.6:                                      ; preds = %for.inc.9.5
  br i1 %tobool7, label %for.inc.9.7, label %for.inc

for.inc.9.7:                                      ; preds = %for.inc.9.6
  br i1 %tobool7, label %for.inc.9.8, label %for.inc

for.inc.9.8:                                      ; preds = %for.inc.9.7
  br i1 %tobool7, label %for.inc.9.9, label %for.inc

for.inc.9.9:                                      ; preds = %for.inc.9.8
  br i1 %tobool7, label %for.inc.9.10, label %for.inc

for.inc.9.10:                                     ; preds = %for.inc.9.9
  br i1 %tobool7, label %for.inc.9.11, label %for.inc

for.inc.9.11:                                     ; preds = %for.inc.9.10
  br i1 %tobool7, label %for.inc.9.12, label %for.inc

for.inc.9.12:                                     ; preds = %for.inc.9.11
  br i1 %tobool7, label %for.inc.9.13, label %for.inc

for.inc.9.13:                                     ; preds = %for.inc.9.12
  br i1 %tobool7, label %for.inc.9.14, label %for.inc

for.inc.9.14:                                     ; preds = %for.inc.9.13
  br i1 %tobool7, label %for.inc.9.15, label %for.inc

for.inc.9.15:                                     ; preds = %for.inc.9.14
  br i1 %tobool7, label %for.inc.9.16, label %for.inc

for.inc.9.16:                                     ; preds = %for.inc.9.15
  br i1 %tobool7, label %for.inc.9.17, label %for.inc

for.inc.9.17:                                     ; preds = %for.inc.9.16
  br i1 %tobool7, label %for.inc.9.18, label %for.inc

for.inc.9.18:                                     ; preds = %for.inc.9.17
  br i1 %tobool7, label %for.inc.9.19, label %for.inc

for.inc.9.19:                                     ; preds = %for.inc.9.18
  br i1 %tobool7, label %for.inc.9.20, label %for.inc

for.inc.9.20:                                     ; preds = %for.inc.9.19
  br i1 %tobool7, label %for.inc.9.21, label %for.inc

for.inc.9.21:                                     ; preds = %for.inc.9.20
  br i1 %tobool7, label %for.inc.9.22, label %for.inc

for.inc.9.22:                                     ; preds = %for.inc.9.21
  br i1 %tobool7, label %for.inc.9.23, label %for.inc

for.inc.9.23:                                     ; preds = %for.inc.9.22
  br i1 %tobool7, label %for.inc.9.24, label %for.inc

for.inc.9.24:                                     ; preds = %for.inc.9.23
  br i1 %tobool7, label %for.inc.9.25, label %for.inc

for.inc.9.25:                                     ; preds = %for.inc.9.24
  br i1 %tobool7, label %for.inc.9.26, label %for.inc

for.inc.9.26:                                     ; preds = %for.inc.9.25
  br i1 %tobool7, label %for.inc.9.27, label %for.inc

for.inc.9.27:                                     ; preds = %for.inc.9.26
  br i1 %tobool7, label %for.inc.9.28, label %for.inc

for.inc.9.28:                                     ; preds = %for.inc.9.27
  br i1 %tobool7, label %for.inc.9.29, label %for.inc

for.inc.9.29:                                     ; preds = %for.inc.9.28
  br i1 %tobool7, label %for.inc.9.30, label %for.inc

for.inc.9.30:                                     ; preds = %for.inc.9.29
  br i1 %tobool7, label %for.inc.9.31, label %for.inc

for.inc.9.31:                                     ; preds = %for.inc.9.30
  br i1 %tobool7, label %for.inc.9.32, label %for.inc

for.inc.9.32:                                     ; preds = %for.inc.9.31
  br i1 %tobool7, label %for.inc.9.33, label %for.inc

for.inc.9.33:                                     ; preds = %for.inc.9.32
  br i1 %tobool7, label %for.inc.9.34, label %for.inc

for.inc.9.34:                                     ; preds = %for.inc.9.33
  br i1 %tobool7, label %for.inc.9.35, label %for.inc

for.inc.9.35:                                     ; preds = %for.inc.9.34
  br i1 %tobool7, label %for.inc.9.36, label %for.inc

for.inc.9.36:                                     ; preds = %for.inc.9.35
  br i1 %tobool7, label %for.inc.9.37, label %for.inc

for.inc.9.37:                                     ; preds = %for.inc.9.36
  br i1 %tobool7, label %for.inc.9.38, label %for.inc

for.inc.9.38:                                     ; preds = %for.inc.9.37
  br i1 %tobool7, label %for.inc.9.39, label %for.inc

for.inc.9.39:                                     ; preds = %for.inc.9.38
  br i1 %tobool7, label %for.inc.9.40, label %for.inc

for.inc.9.40:                                     ; preds = %for.inc.9.39
  br i1 %tobool7, label %for.inc.9.41, label %for.inc

for.inc.9.41:                                     ; preds = %for.inc.9.40
  br i1 %tobool7, label %for.inc.9.42, label %for.inc

for.inc.9.42:                                     ; preds = %for.inc.9.41
  br i1 %tobool7, label %for.inc.9.43, label %for.inc

for.inc.9.43:                                     ; preds = %for.inc.9.42
  br i1 %tobool7, label %if.end.12, label %for.inc
}
