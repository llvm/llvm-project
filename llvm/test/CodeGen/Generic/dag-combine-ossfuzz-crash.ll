; RUN: llc < %s

; llc built with address sanitizer crashes because of a dangling node pointer
; oss-fuzz -  DAGCombiner::useDivRem (5011)

define void @f() {
BB:
  %A19 = alloca i1**
  %C18 = icmp ugt i1 true, false
  %L13 = load i8, i8* undef
  %L10 = load i8, i8* undef
  %B12 = and i8 %L13, %L10
  %B35 = and i8 %B12, %L10
  %L2 = load i66*, i66** undef
  %L23 = load i66, i66* %L2
  %B38 = urem i8 %B35, %B12
  %B9 = ashr i66 %L23, %L23
  %C11 = icmp sge i8 %B38, %B35
  %A4 = alloca i66
  %G4 = getelementptr i66, i66* %A4, i1 true
  %L6 = load i66, i66* %G4
  %B21 = urem i1 %C11, true
  %B1 = mul i66 %B9, %L23
  %B5 = udiv i8 %L10, %L13
  %B22 = udiv i66 %B9, %B1
  %C29 = icmp ult i32 -1, 0
  store i1* undef, i1** undef
  store i1 %C29, i1* undef
  br label %BB1

BB1:                                              ; preds = %BB
  %G8 = getelementptr i66, i66* undef, i16 32767
  %G43 = getelementptr i66, i66* undef, i66 -1
  %L20 = load i1, i1* undef
  %B7 = and i66 %L6, %L6
  %B30 = sdiv i66 -36893488147419103232, -1
  %B16 = urem i66 %B22, %L6
  %G47 = getelementptr i66, i66* %G8, i66 %B16
  store i66 %B7, i66* %G47
  store i8 %B5, i8* undef
  %C5 = icmp ult i1 %C18, %L20
  store i66 %B30, i66* %G47
  store i1** undef, i1*** %A19
  store i1 %C5, i1* undef
  store i1 %C11, i1* undef
  store i66* %G43, i66** undef
  store i1 %B21, i1* undef
  %G59 = getelementptr i1, i1* undef, i1 false
  %G61 = getelementptr i66, i66* %G8, i1 %L20
  store i1 %L20, i1* %G59
  store i66* %G61, i66** undef
  ret void
}
