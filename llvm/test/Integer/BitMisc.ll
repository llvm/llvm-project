; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


@MyVar     = external global i19
@MyIntList = external global { i39 *, i19 }
             external global i19      ; i19*:0

@AConst    = constant i19 -123

@AString   = constant [4 x i8] c"test"

@ZeroInit  = global { [100 x i19 ], [40 x float ] } { [100 x i19] zeroinitializer,
                                                      [40  x float] zeroinitializer }


define i19 @"foo"(i19 %blah)
begin
	store i19 5, i19* @MyVar
	%idx = getelementptr { i39 *, i19 } * @MyIntList, i64 0, i32 1
  	store i19 12, i19* %idx
  	ret i19 %blah
end
