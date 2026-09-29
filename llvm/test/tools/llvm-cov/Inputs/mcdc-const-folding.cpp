#include <stdlib.h>

bool case0(bool a) {
    return 0 && a;
}
bool case1(bool a) {
    return a && 0;
}
bool case2(bool a) {
    return 1 && a;
}
bool case3(bool a) {
    return a && 1;
}
bool case4(bool a) {
    return 1 || a;
}
bool case5(bool a) {
    return a || 1;
}
bool case6(bool a) {
    return 0 || a;
}
bool case7(bool a) {
    return a || 0;
}

bool case8(bool a, bool b) {
    return 0 && a && b;
}
bool case9(bool a, bool b) {
    return a && 0 && b;
}
bool casea(bool a, bool b) {
    return 1 && a && b;
}
bool caseb(bool a, bool b) {
    return a && 1 && b;
}
bool casec(bool a, bool b) {
    return 1 || a || b;
}
bool cased(bool a, bool b) {
    return a || 1 || b;
}
bool casee(bool a, bool b) {
    return 0 || a || b;
}
bool casef(bool a, bool b) {
    return a || 0 || b;
}

bool caseg(bool a, bool b) {
    return b && a && 0;
}
bool caseh(bool a, bool b) {
    return b && 0 && a;
}
bool casei(bool a, bool b) {
    return b && a && 1;
}
bool casej(bool a, bool b) {
    return b && 1 && a;
}
bool casek(bool a, bool b) {
    return b || a || 1;
}
bool casel(bool a, bool b) {
    return b || 1 || a;
}
bool casem(bool a, bool b) {
    return b || a || 0;
}
bool casen(bool a, bool b) {
    return b || 0 || a;
}

int main(int argc, char *argv[])
{
    bool a = atoi(argv[1]);
    bool b = atoi(argv[2]);
    volatile bool c;

    c = case0(a);
    c = case1(a);
    c = case2(a);
    c = case3(a);
    c = case4(a);
    c = case5(a);
    c = case6(a);
    c = case7(a);

    c = case8(a, b);
    c = case9(a, b);
    c = casea(a, b);
    c = caseb(a, b);
    c = casec(a, b);
    c = cased(a, b);
    c = casee(a, b);
    c = casef(a, b);

    c = caseg(a, b);
    c = caseh(a, b);
    c = casei(a, b);
    c = casej(a, b);
    c = casek(a, b);
    c = casel(a, b);
    c = casem(a, b);
    c = casen(a, b);

    return 0;
}
