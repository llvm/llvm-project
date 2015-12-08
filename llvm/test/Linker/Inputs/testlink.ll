%intlist = type { %intlist*, i32 }


%Ty1 = type { %Ty2* }
%Ty2 = type opaque

%VecSize = type { <10 x i32> }

@GVTy1 = global %Ty1* null
@GVTy2 = external global %Ty2*


@MyVar = global i32 4
@MyIntList = external global %intlist
@AConst = constant i32 1234

;; Intern in both testlink[12].ll
@Intern1 = internal constant i32 52

@Use2Intern1 = global i32* @Intern1

;; Intern in one but not in other
@Intern2 = constant i32 12345

@MyIntListPtr = constant { %intlist* } { %intlist* @MyIntList }
@MyVarPtr = linkonce global { i32* } { i32* @MyVar }
@0 = constant i32 412

; Provides definition of Struct1 and of S1GV.
%Struct1 = type { i32 }
@S1GV = global %Struct1* null

define i32 @foo(i32 %blah) {
  store i32 %blah, i32* @MyVar
  %idx = getelementptr %intlist, %intlist* @MyIntList, i64 0, i32 1
  store i32 12, i32* %idx
  %ack = load i32, i32* @0
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
