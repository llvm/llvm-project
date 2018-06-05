// Compile with "cl /c /ZI /sdl /EHsc /MTd /permissive-
// test-pdb-function-level-linking.cpp"
// Link with "link /debug:full test-pdb-function-level-linking.obj"

#include <memory>
#include <string>

std::string foo()
{
    return "Hello!";
}

int main()
{
    auto x = foo();
    return 0;
}
