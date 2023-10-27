; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -debug-pass=Structure -enable-fs-discriminator=true -improved-fs-discriminator=true 2>&1 | FileCheck %s --check-prefix=NOPROFILE
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -debug-pass=Structure -fs-profile-file=%S/Inputs/fsloader-mfs.afdo -enable-fs-discriminator=true -improved-fs-discriminator=true 2>&1 | FileCheck %s --check-prefix=PROFILE-NOMFS
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -debug-pass=Structure -fs-profile-file=%S/Inputs/fsloader-mfs.afdo -split-machine-functions -enable-fs-discriminator=true -improved-fs-discriminator=true 2>&1 | FileCheck %s --check-prefix=PROFILE-MFS

;; No profile is specified, no load passes.
; NOPROFILE: Add FS discriminators in MIR
; NO-NOPROFILE: SampleFDO loader in MIR
; NOPROFILE: Add FS discriminators in MIR
; NO-NOPROFILE: SampleFDO loader in MIR
; NOPROFILE: Add FS discriminators in MIR
; NO-NOPROFILE: SampleFDO loader in MIR

;; Profile is specified, so we have first 2 load passes.
; PROFILE-NOMFS: Add FS discriminators in MIR
; PROFILE-NOMFS: SampleFDO loader in MIR
; PROFILE-NOMFS: Add FS discriminators in MIR
; PROFILE-NOMFS: SampleFDO loader in MIR
; PROFILE-NOMFS: Add FS discriminators in MIR
;; But mfs is not specified, so no "SampleFDO loader should be created"
; NO-PROFILE-NOMFS: SampleFDO loader in MIR

;; Profile is specified with mfs, so we have 3 load passes.
; PROFILE-MFS: Add FS discriminators in MIR
; PROFILE-MFS: SampleFDO loader in MIR
; PROFILE-MFS: Add FS discriminators in MIR
; PROFILE-MFS: SampleFDO loader in MIR
; PROFILE-MFS: Add FS discriminators in MIR
; PROFILE-MFS: SampleFDO loader in MIR
; PROFILE-MFS: Machine Function Splitter Transformation

define void @foo4(i1 zeroext %0, i1 zeroext %1) nounwind {
  br i1 %0, label %3, label %7

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = tail call i32 @qux()
  ret void
}

declare i32 @bar()
declare i32 @baz()
declare i32 @bam()
declare i32 @qux()
