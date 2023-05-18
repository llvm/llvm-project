// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not llvm-tblgen -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not llvm-tblgen -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not llvm-tblgen -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// RUN: not llvm-tblgen -DERROR8 %s 2>&1 | FileCheck --check-prefix=ERROR8 %s

class TestClass<int a, int b = 2, int c = 3> {
  int value =  !add(a, b, c);
}
// CHECK:      def testClass1 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass2 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass3 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass4 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass5 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass6 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass7 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testClass8 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
def testClass1: TestClass<1>;
def testClass2: TestClass<1, 2>;
def testClass3: TestClass<1, 2, 3>;
def testClass4: TestClass<1, b=2>;
def testClass5: TestClass<1, c=3>;
def testClass6: TestClass<1, b=2, c=3>;
def testClass7: TestClass<1, c=3, b=2>;
def testClass8: TestClass<a=1, c=3, b=2>;

multiclass TestMultiClass<int a, int b = 2, int c = 3> {
  def "": TestClass<a, b=b, c=c>;
}

// CHECK:      def testMultiClass1 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass2 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass3 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass4 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass5 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass6 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass7 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
// CHECK:      def testMultiClass8 {
// CHECK-NEXT:   int value = 6;
// CHECK-NEXT: }
defm testMultiClass1: TestMultiClass<1>;
defm testMultiClass2: TestMultiClass<1, 2>;
defm testMultiClass3: TestMultiClass<1, 2, 3>;
defm testMultiClass4: TestMultiClass<1, b=2>;
defm testMultiClass5: TestMultiClass<1, c=3>;
defm testMultiClass6: TestMultiClass<1, b=2, c=3>;
defm testMultiClass7: TestMultiClass<1, c=3, b=2>;
defm testMultiClass8: TestMultiClass<a=1, b=2, c=3>;

class TestSubroutine<int a, int b=a>{
  int value=!add(a, b);
}

// CHECK:      def testSubroutine {
// CHECK-NEXT:   int value1 = 2;
// CHECK-NEXT:   int value2 = 2;
// CHECK-NEXT:   int value3 = 2;
// CHECK-NEXT: }
def testSubroutine {
  int value1=TestSubroutine<1>.value;
  int value2=TestSubroutine<1, b=1>.value;
  int value3=TestSubroutine<b=1, a=1>.value;
}

#ifdef ERROR1
// ERROR1: Argument "d" doesn't exist
def testError1: TestClass<1, d=3>;
#endif

#ifdef ERROR2
// ERROR2: The name of named argument should be a valid identifier
def testError2: TestClass<1, 3=0>;
#endif

#ifdef ERROR3
// ERROR3: Positional argument should be put before named argument
def testError3: TestClass<1, b=1, 2>;
#endif

#ifdef ERROR4
// ERROR4: The value of named argument should be initialized, but we got '?'
def testError4: TestClass<1, b=?>;
#endif

#ifdef ERROR5
// ERROR5: We can only specify the template argument 'TestClass:a' once
def testError5: TestClass<1, a=1>;
#endif

#ifdef ERROR6
// ERROR6: We can only specify the template argument 'TestMultiClass::a' once
defm testError6: TestMultiClass<1, a=1>;
#endif

#ifdef ERROR7
// ERROR7: We can only specify the template argument 'TestSubroutine:a' once
def testError7 {
  int value=TestSubroutine<1, a=1>.value;
}
#endif

#ifdef ERROR8
// ERROR8: We can only specify the template argument 'TestClass:b' once
def testError8: TestClass<a=1, b=1, b=1>;
#endif
