typedef int Foo;

int main() {
    Foo array[3] = {1,2,3};
    return 0; //% self.expect("frame variable array --show-types --", substrs = ['(Foo [3]) wrong_type_here = {','(Foo) [0] = 1','(Foo) [1] = 2','(Foo) [2] = 3'])
}
