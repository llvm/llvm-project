#include <vector>
#include <string>

void f() {}

int main() {
    int foo = 10;
    int *bar = new int(4);
    std::string baz = "85";
    
    f(); // break here
    f(); // break here
    return 0;
}
