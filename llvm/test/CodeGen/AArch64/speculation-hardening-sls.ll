; RUN: llc -mattr=harden-sls-retbr -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,ISBDSB
; RUN: llc -mattr=harden-sls-retbr -mattr=+sb -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,SB


; Function Attrs: norecurse nounwind readnone
define dso_local i32 @double_return(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %div = sdiv i32 %a, %b
  ret i32 %div

if.else:                                          ; preds = %entry
  %div1 = sdiv i32 %b, %a
  ret i32 %div1
; CHECK-LABEL: double_return:
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
}

@__const.indirect_branch.ptr = private unnamed_addr constant [2 x i8*] [i8* blockaddress(@indirect_branch, %return), i8* blockaddress(@indirect_branch, %l2)], align 8

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @indirect_branch(i32 %a, i32 %b, i32 %i) {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [2 x i8*], [2 x i8*]* @__const.indirect_branch.ptr, i64 0, i64 %idxprom
  %0 = load i8*, i8** %arrayidx, align 8
  indirectbr i8* %0, [label %return, label %l2]

l2:                                               ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %l2
  %retval.0 = phi i32 [ 1, %l2 ], [ 0, %entry ]
  ret i32 %retval.0
; CHECK-LABEL: indirect_branch:
; CHECK:       br x
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
}

; Check that RETAA and RETAB instructions are also protected as expected.
define dso_local i32 @ret_aa(i32 returned %a) local_unnamed_addr "target-features"="+neon,+v8.3a" "sign-return-address"="all" "sign-return-address-key"="a_key" {
entry:
; CHECK-LABEL: ret_aa:
; CHECK:       {{ retaa$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
	  ret i32 %a
}

define dso_local i32 @ret_ab(i32 returned %a) local_unnamed_addr "target-features"="+neon,+v8.3a" "sign-return-address"="all" "sign-return-address-key"="b_key" {
entry:
; CHECK-LABEL: ret_ab:
; CHECK:       {{ retab$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
	  ret i32 %a
}

define i32 @asmgoto() {
entry:
; CHECK-LABEL: asmgoto:
  callbr void asm sideeffect "B $0", "X"(i8* blockaddress(@asmgoto, %d))
            to label %asm.fallthrough [label %d]
     ; The asm goto above produces a direct branch:
; CHECK:           //APP
; CHECK-NEXT:      {{^[ \t]+b }}
; CHECK-NEXT:      //NO_APP
     ; For direct branches, no mitigation is needed.
; ISDDSB-NOT: dsb sy
; SB-NOT:     {{ sb$}}

asm.fallthrough:               ; preds = %entry
  ret i32 0
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

d:                             ; preds = %asm.fallthrough, %entry
  ret i32 1
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}
