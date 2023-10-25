; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu < %s \
; RUN: -mcpu=pwr8 -mattr=+regnames | FileCheck --check-prefix=FULLNAMES %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-linux-gnu < %s \
; RUN: -mcpu=pwr8 -mattr=+regnames | FileCheck --check-prefix=FULLNAMES %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s \
; RUN: -mcpu=pwr8 -mattr=+regnames | FileCheck --check-prefix=FULLNAMES %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu < %s \
; RUN: -mcpu=pwr8 -mattr=-regnames | FileCheck --check-prefix=NOFULLNAMES %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-linux-gnu < %s \
; RUN: -mcpu=pwr8 -mattr=-regnames | FileCheck --check-prefix=NOFULLNAMES %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s \
; RUN: -mcpu=pwr8 -mattr=-regnames | FileCheck --check-prefix=NOFULLNAMES %s


define dso_local signext i32 @IntNames(i32 noundef signext %a, i32 noundef signext %b) local_unnamed_addr #0 {
; FULLNAMES-LABEL: IntNames:
; FULLNAMES:       # %bb.0: # %entry
; FULLNAMES-NEXT:    add r3, r4, r3
; FULLNAMES-NEXT:    extsw r3, r3
; FULLNAMES-NEXT:    blr
;
; NOFULLNAMES-LABEL: IntNames:
; NOFULLNAMES:       # %bb.0: # %entry
; NOFULLNAMES-NEXT:    add 3, 4, 3
; NOFULLNAMES-NEXT:    extsw 3, 3
; NOFULLNAMES-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define dso_local double @FPNames(double noundef %a, double noundef %b) local_unnamed_addr #0 {
; FULLNAMES-LABEL: FPNames:
; FULLNAMES:       # %bb.0: # %entry
; FULLNAMES-NEXT:    xsadddp f1, f1, f2
; FULLNAMES-NEXT:    blr
;
; NOFULLNAMES-LABEL: FPNames:
; NOFULLNAMES:       # %bb.0: # %entry
; NOFULLNAMES-NEXT:    xsadddp 1, 1, 2
; NOFULLNAMES-NEXT:    blr
entry:
  %add = fadd double %a, %b
  ret double %add
}

define dso_local <4 x float> @VecNames(<4 x float> noundef %a, <4 x float> noundef %b) local_unnamed_addr #0 {
; FULLNAMES-LABEL: VecNames:
; FULLNAMES:       # %bb.0: # %entry
; FULLNAMES-NEXT:    xvaddsp vs34, vs34, vs35
; FULLNAMES-NEXT:    blr
;
; NOFULLNAMES-LABEL: VecNames:
; NOFULLNAMES:       # %bb.0: # %entry
; NOFULLNAMES-NEXT:    xvaddsp 34, 34, 35
; NOFULLNAMES-NEXT:    blr
entry:
  %add = fadd <4 x float> %a, %b
  ret <4 x float> %add
}

attributes #0 = { nounwind willreturn "target-features"="+altivec" }
