; RUN: llc -mtriple=thumbv7-apple-darwin10 -arm-atomic-cfg-tidy=0 < %s | FileCheck %s

%struct.op = type { %struct.op*, %struct.op*, %struct.op* ()*, i32, i16, i16, i8, i8 }

; CHECK: Perl_ck_sort
; CHECK: ldr
; CHECK: mov [[REGISTER:(r[0-9]+)|(lr)]]
; CHECK: str {{(r[0-9])|(lr)}}, [[[REGISTER]]], #24

define void @Perl_ck_sort() nounwind optsize {
entry:
  %tmp27 = load %struct.op*, %struct.op** undef, align 4
  switch i16 undef, label %if.end151 [
    i16 178, label %if.then60
    i16 177, label %if.then60
  ]

if.then60:                                        ; preds = %if.then40
  br i1 undef, label %if.then67, label %if.end95

if.then67:                                        ; preds = %if.then60
  %op_next71 = getelementptr inbounds %struct.op, %struct.op* %tmp27, i32 0, i32 0
  store %struct.op* %tmp27, %struct.op** %op_next71, align 4
  %0 = getelementptr inbounds %struct.op, %struct.op* %tmp27, i32 1, i32 0
  br label %if.end95

if.end95:                                         ; preds = %if.else92, %if.then67
  %.pre-phi = phi %struct.op** [ undef, %if.then60 ], [ %0, %if.then67 ]
  %tmp98 = load %struct.op*, %struct.op** %.pre-phi, align 4
  br label %if.end151

if.end151:                                        ; preds = %if.end100, %if.end, %entry
  ret void
}
