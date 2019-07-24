; RUN: llvm-as < %s -o /dev/null

define i8* @f_0(i8 %val) {
  %ptr = inttoptr i8 %val to i8*, !dereferenceable_or_null !{i64 2}
  ret i8* %ptr 
}
