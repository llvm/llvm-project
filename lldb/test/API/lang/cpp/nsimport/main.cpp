namespace N
{
    int n;
}

namespace
{
    int anon;
}

namespace Nested
{
    namespace
    {
        int nested;
    }
}

namespace Global
{
    int global;
}

namespace Fun
{
    int fun_var;
    int fun()
    {
        fun_var = 5;
        return 0; // break 1
    }
}

namespace Single
{
    int single = 3;
}

namespace NotImportedBefore
{
    int not_imported = 45;
}

using namespace Global;

int not_imported = 35;
int fun_var = 9;

namespace NotImportedAfter
{
    int not_imported = 55;
}

namespace Imported
{
    int imported = 99;
}

int imported = 89;

int main()
{
    using namespace N;
    using namespace Nested;
    using namespace Imported;
    using Single::single;
    n = 1;
    anon = 2;
    nested = 3;
    global = 4;
    return Fun::fun(); // break 0
}
