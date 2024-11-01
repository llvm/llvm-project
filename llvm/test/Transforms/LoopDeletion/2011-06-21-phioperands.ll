; RUN: opt -passes=loop-deletion -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%0 = type { %"class.llvm::SmallVectorImpl", [1 x %"union.llvm::SmallVectorBase::U"] }
%"class.clang::SourceLocation" = type { i32 }
%"class.clang::driver::Arg" = type { %"class.clang::driver::Option"*, %"class.clang::driver::Arg"*, i32, i8, %0 }
%"class.clang::driver::Option" = type { i32 (...)**, i32, %"class.clang::SourceLocation", i8*, %"class.clang::driver::OptionGroup"*, %"class.clang::driver::Option"*, i8 }
%"class.clang::driver::OptionGroup" = type { %"class.clang::driver::Option" }
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8*, %"union.llvm::SmallVectorBase::U" }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase" }
%"class.llvm::SmallVectorTemplateBase" = type { %"class.llvm::SmallVectorTemplateCommon" }
%"class.llvm::SmallVectorTemplateCommon" = type { %"class.llvm::SmallVectorBase" }
%"union.llvm::SmallVectorBase::U" = type { x86_fp80 }

define void @_ZNK5clang6driver7ArgList20AddAllArgsTranslatedERN4llvm11SmallVectorIPKcLj16EEENS0_12OptSpecifierES5_b(i1 zeroext %Joined) nounwind align 2 {
entry:
  br i1 undef, label %entry.split.us, label %entry.entry.split_crit_edge

entry.entry.split_crit_edge:                      ; preds = %entry
  br label %entry.split

entry.split.us:                                   ; preds = %entry
  br label %for.cond.i14.us

for.cond.i14.us:                                  ; preds = %for.inc.i38.us, %entry.split.us
  br i1 true, label %for.cond.i50.us-lcssa.us, label %if.end.i23.us

for.inc.i38.us:                                   ; preds = %if.end.i23.us
  br label %for.cond.i14.us

if.end.i23.us:                                    ; preds = %for.cond.i14.us
  br i1 true, label %for.cond.i50.us-lcssa.us, label %for.inc.i38.us

for.cond.i50.us-lcssa.us:                         ; preds = %if.end.i23.us, %for.cond.i14.us
  br label %for.cond.i50

entry.split:                                      ; preds = %entry.entry.split_crit_edge
  br label %for.cond.i14

for.cond.i14:                                     ; preds = %for.inc.i38, %entry.split
  br i1 undef, label %for.cond.i50.us-lcssa, label %if.end.i23

if.end.i23:                                       ; preds = %for.cond.i14
  br i1 undef, label %for.cond.i50.us-lcssa, label %for.inc.i38

for.inc.i38:                                      ; preds = %if.end.i23
  br label %for.cond.i14

for.cond.i50.us-lcssa:                            ; preds = %if.end.i23, %for.cond.i14
  br label %for.cond.i50

for.cond.i50:                                     ; preds = %for.cond.i50.us-lcssa, %for.cond.i50.us-lcssa.us
  br label %for.cond

for.cond.loopexit.us-lcssa:                       ; preds = %if.end.i, %for.cond.i
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.cond.loopexit.us-lcssa.us, %for.cond.loopexit.us-lcssa
  br label %for.cond

for.cond:                                         ; preds = %for.cond.loopexit, %for.cond.i50
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  br i1 %Joined, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  br i1 undef, label %cond.false.i.i, label %_ZN4llvm9StringRefC1EPKc.exit

cond.false.i.i:                                   ; preds = %if.then
  unreachable

_ZN4llvm9StringRefC1EPKc.exit:                    ; preds = %if.then
  br i1 undef, label %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit, label %cond.false.i.i91

cond.false.i.i91:                                 ; preds = %_ZN4llvm9StringRefC1EPKc.exit
  unreachable

_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit: ; preds = %_ZN4llvm9StringRefC1EPKc.exit
  br i1 undef, label %cond.false.i.i.i, label %if.end13.i.i.i.i

if.end13.i.i.i.i:                                 ; preds = %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit
  br i1 undef, label %land.lhs.true16.i.i.i.i, label %if.end19.i.i.i.i

land.lhs.true16.i.i.i.i:                          ; preds = %if.end13.i.i.i.i
  br i1 undef, label %cond.false.i.i.i, label %_ZNK4llvm5Twine8isBinaryEv.exit8.i.i.i.i

_ZNK4llvm5Twine8isBinaryEv.exit8.i.i.i.i:         ; preds = %land.lhs.true16.i.i.i.i
  br i1 undef, label %cond.false.i.i.i, label %if.end19.i.i.i.i

if.end19.i.i.i.i:                                 ; preds = %_ZNK4llvm5Twine8isBinaryEv.exit8.i.i.i.i, %if.end13.i.i.i.i
  br i1 undef, label %land.lhs.true22.i.i.i.i, label %_ZN4llvmplERKNS_9StringRefEPKc.exit

land.lhs.true22.i.i.i.i:                          ; preds = %if.end19.i.i.i.i
  br i1 undef, label %cond.false.i.i.i, label %_ZNK4llvm5Twine8isBinaryEv.exit.i.i.i.i

_ZNK4llvm5Twine8isBinaryEv.exit.i.i.i.i:          ; preds = %land.lhs.true22.i.i.i.i
  br i1 undef, label %cond.false.i.i.i, label %_ZN4llvmplERKNS_9StringRefEPKc.exit

cond.false.i.i.i:                                 ; preds = %_ZNK4llvm5Twine8isBinaryEv.exit.i.i.i.i, %land.lhs.true22.i.i.i.i, %_ZNK4llvm5Twine8isBinaryEv.exit8.i.i.i.i, %land.lhs.true16.i.i.i.i, %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit
  unreachable

_ZN4llvmplERKNS_9StringRefEPKc.exit:              ; preds = %_ZNK4llvm5Twine8isBinaryEv.exit.i.i.i.i, %if.end19.i.i.i.i
  br i1 undef, label %Retry.i, label %if.end.i99

Retry.i:                                          ; preds = %if.end.i99, %_ZN4llvmplERKNS_9StringRefEPKc.exit
  br i1 undef, label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit, label %new.notnull.i

new.notnull.i:                                    ; preds = %Retry.i
  br label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit

if.end.i99:                                       ; preds = %_ZN4llvmplERKNS_9StringRefEPKc.exit
  br label %Retry.i

_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit: ; preds = %new.notnull.i, %Retry.i
  br label %for.cond.i.preheader

if.else:                                          ; preds = %for.body
  br i1 undef, label %Retry.i108, label %if.end.i113

Retry.i108:                                       ; preds = %if.end.i113, %if.else
  br i1 undef, label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit114, label %new.notnull.i110

new.notnull.i110:                                 ; preds = %Retry.i108
  br label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit114

if.end.i113:                                      ; preds = %if.else
  br label %Retry.i108

_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit114: ; preds = %new.notnull.i110, %Retry.i108
  br i1 undef, label %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit125, label %cond.false.i.i123

cond.false.i.i123:                                ; preds = %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit114
  unreachable

_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit125: ; preds = %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit114
  br i1 undef, label %Retry.i134, label %if.end.i139

Retry.i134:                                       ; preds = %if.end.i139, %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit125
  br i1 undef, label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit140, label %new.notnull.i136

new.notnull.i136:                                 ; preds = %Retry.i134
  br label %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit140

if.end.i139:                                      ; preds = %_ZNK5clang6driver3Arg8getValueERKNS0_7ArgListEj.exit125
  br label %Retry.i134

_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit140: ; preds = %new.notnull.i136, %Retry.i134
  br label %for.cond.i.preheader

for.cond.i.preheader:                             ; preds = %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit140, %_ZN4llvm15SmallVectorImplIPKcE9push_backERKS2_.exit
  br i1 undef, label %for.cond.i.preheader.split.us, label %for.cond.i.preheader.for.cond.i.preheader.split_crit_edge

for.cond.i.preheader.for.cond.i.preheader.split_crit_edge: ; preds = %for.cond.i.preheader
  br label %for.cond.i.preheader.split

for.cond.i.preheader.split.us:                    ; preds = %for.cond.i.preheader
  br label %for.cond.i.us

for.cond.i.us:                                    ; preds = %if.end.i.us, %for.cond.i.preheader.split.us
  br i1 true, label %for.cond.loopexit.us-lcssa.us, label %if.end.i.us

if.end.i.us:                                      ; preds = %for.cond.i.us
  br i1 true, label %for.cond.loopexit.us-lcssa.us, label %for.cond.i.us

for.cond.loopexit.us-lcssa.us:                    ; preds = %if.end.i.us, %for.cond.i.us
  %tmp178218.us.lcssa = phi %"class.clang::driver::Arg"** [ undef, %if.end.i.us ], [ undef, %for.cond.i.us ]
  br label %for.cond.loopexit

for.cond.i.preheader.split:                       ; preds = %for.cond.i.preheader.for.cond.i.preheader.split_crit_edge
  br label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i, %for.cond.i.preheader.split
  br i1 undef, label %for.cond.loopexit.us-lcssa, label %if.end.i

if.end.i:                                         ; preds = %for.cond.i
  br i1 undef, label %for.cond.loopexit.us-lcssa, label %for.cond.i

for.end:                                          ; preds = %for.cond
  ret void
}
