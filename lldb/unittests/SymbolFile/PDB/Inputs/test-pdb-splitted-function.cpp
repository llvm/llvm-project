// Compile with "cl /c /Zi /GL /O2 /EHsc /MTd test-pdb-splitted-function.cpp"
// Link with "link /debug:full /LTCG /GENPROFILE
//   test-pdb-splitted-function.obj"
// Run several times
// Link with "link /debug:full /LTCG /USEPROFILE
//   test-pdb-splitted-function.obj"

#include <cmath>
#include <iostream>

int main()
{
    auto b = false;
    for (auto i = 1; i <= 1024; i++)
    {
        if (b)
        {
            std::cout << "Unreachable code" << std::endl;
            auto x = std::sin(i);
            return x;
        }

        b = (i % 2 + (i - 1) % 2) != 1;
    }

    return 0;
}
