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

namespace aa {
    class cc {
    public:
        cc();
        ~cc();
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

    cc::cc() {}
    cc::~cc() {}
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
    aa::cc aac;
    b::c bc;
    c::d cd;
    ac.func1();
    ac.func2();
    ac.func3();
    aac.func1();
    aac.func2();
    aac.func3();
    bc.func1();
    bc.func3();
    cd.func2();
    cd.func3();
    return 0;
}
