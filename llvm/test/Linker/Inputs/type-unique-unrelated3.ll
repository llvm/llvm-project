%t = type { ptr }
declare %t @f()

define %t @g() {
 %x = call %t @f()
 ret %t %x
}
