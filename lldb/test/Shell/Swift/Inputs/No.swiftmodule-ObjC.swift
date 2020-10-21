import Foundation
import ObjCStuff

func use<T>(_ t: T) {}

func f() {
  // CHECK-DAG: (size_t) ctype = 1024
  let ctype = size_t(1024)
  // This works as a Clang type via the Objective-C runtime.
  // CHECK-DAG: object = 0x{{[0-9a-f]+}} Int32(1234)
  // FIXME: (ObjCClass) object = {{.*}}Hello from Objective-C!
  let object = ObjCClass()
  // The Objective-C runtime recognizes this as a tagged pointer.
  // CHECK-DAG: (NSNumber) inlined = {{.*}}Int64(42)
  let inlined = NSNumber(value: 42)
  // CHECK-DAG: (CMYK) enumerator = .yellow
  let enumerator = yellow
  // CHECK-DAG: (FourColors) typedef = .cyan
  let typedef = FourColors(0)
  let union = Union(i: 23)
  // CHECK-DAG: (OBJCSTUFF_MyString) renamed = {{.*}} "with swift_name"
  let renamed = MyString("with swift_name")
  // CHECK-DAG: (MyFloat) globalFloat = 3.14
  use(ctype) // break here
  use(object)
  use(inlined)
  use(enumerator)
  use(typedef)
  use(union)
  use(renamed)
}

f()
