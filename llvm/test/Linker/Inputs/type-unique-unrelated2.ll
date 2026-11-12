%t = type { ptr }
declare %t @g()

define %t @g2() {
 %x = call %t @g()
 ret %t %x
}
