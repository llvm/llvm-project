#include <stdio.h>
#include <stdint.h>

namespace a {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func2() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace b {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace c {
    class d {
    public:
        d () {}
        ~d() {}
        void func2() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };
}

int main (int argc, char const *argv[])
{
    a::c ac;
    b::c bc;
    c::d cd;
    ac.func1();
    ac.func2();
    ac.func3();
    bc.func1();
    bc.func3();
    cd.func2();
    cd.func3();
    return 0;
}
