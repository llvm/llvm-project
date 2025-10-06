extern void subroutine_foo(void);
extern void subroutine_bar(void);

void func_a(int x, int y) {
    if (x == 0 || y == 0)
        subroutine_foo();
    else
        subroutine_bar();
}

void func_b(int x, int y) {
    if (x == 0)
        subroutine_foo();
    else if (y == 0)
        subroutine_foo();
    else
        subroutine_bar();
}

