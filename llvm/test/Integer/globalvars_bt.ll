; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll



@MyVar     = external global i27
@MyIntList = external global { \2 *, i27 }
             external global i27      ; i27*:0

@AConst    = constant i27 123

@AString   = constant [4 x i8] c"test"

@ZeroInit  = global { [100 x i27 ], [40 x float ] } { [100 x i27] zeroinitializer,
                                                      [40  x float] zeroinitializer }


define i27 @"foo"(i27 %blah)
begin
	store i27 5, i27 *@MyVar
        %idx = getelementptr { \2 *, i27 } * @MyIntList, i64 0, i32 1
  	store i27 12, i27* %idx
  	ret i27 %blah
end

