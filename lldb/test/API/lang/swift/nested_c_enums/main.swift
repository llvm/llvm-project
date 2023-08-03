public enum Enum1 {
    case A
    case B
}
public enum Enum2 {
    case C
    case D
}
public enum SuperEnum {
    case Case1(Enum1)
    case Case2(Enum2)
}
let x = SuperEnum.Case1(.A)
let y = SuperEnum.Case1(.B)
let w = SuperEnum.Case2(.C)
let z = SuperEnum.Case2(.D)
print() //%self.expect("frame var -d run-target -- x", substrs=['(a.SuperEnum)','Case1 = A'])
        //%self.expect("expr -d run-target -- x", substrs=['(a.SuperEnum)','Case1 = A'])
        //%self.expect("frame var -d run-target -- y", substrs=['(a.SuperEnum)','Case1 = B'])
        //%self.expect("expr -d run-target -- y", substrs=['(a.SuperEnum)','Case1 = B'])
        //%self.expect("frame var -d run-target -- w", substrs=['(a.SuperEnum)','Case2 = C'])
        //%self.expect("expr -d run-target -- w", substrs=['(a.SuperEnum)','Case2 = C'])
        //%self.expect("frame var -d run-target -- z", substrs=['(a.SuperEnum)','Case2 = D'])
        //%self.expect("expr -d run-target -- z", substrs=['(a.SuperEnum)','Case2 = D'])
