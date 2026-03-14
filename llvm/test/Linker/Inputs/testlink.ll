%intlist = type { ptr, i32 }


%Ty1 = type { ptr }
%Ty2 = type opaque

%VecSize = type { <10 x i32> }

@GVTy1 = global %Ty1 { ptr null }
@GVTy2 = external global %Ty2


@MyVar = global i32 4
@MyIntList = external global %intlist
@AConst = constant i32 1234

;; Intern in both testlink[12].ll
@Intern1 = internal constant i32 52

@Use2Intern1 = global ptr @Intern1

;; Intern in one but not in other
@Intern2 = constant i32 12345

@MyIntListPtr = constant { ptr } { ptr @MyIntList }
@MyVarPtr = linkonce global { ptr } { ptr @MyVar }
@0 = constant i32 412

; Provides definition of Struct1 and of S1GV.
%Struct1 = type { i32 }
@S1GV = global ptr null

define i32 @foo(i32 %blah) {
  store i32 %blah, ptr @MyVar
  %idx = getelementptr %intlist, ptr @MyIntList, i64 0, i32 1
  store i32 12, ptr %idx
  %ack = load i32, ptr @0
  %fzo = add i32 %ack, %blah
  ret i32 %fzo
}

declare void @unimp(float, double)

define internal void @testintern() {
  ret void
}

define void @Testintern() {
  ret void
}

define internal void @testIntern() {
  ret void
}

define void @VecSizeCrash1(%VecSize) {
  ret void
}
