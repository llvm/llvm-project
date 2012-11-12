#include <string>

struct DeepData_5
{
    std::string m_some_text;
    DeepData_5() :
    m_some_text("Just a test") {}
};

struct DeepData_4
{
    DeepData_5 m_child1;
    DeepData_5 m_child2;
    DeepData_5 m_child3;
};

struct DeepData_3
{
    DeepData_4& m_child1;
    DeepData_4 m_child2;
    
    DeepData_3() : m_child1(* (new DeepData_4())), m_child2(DeepData_4()) {}
};

struct DeepData_2
{
    DeepData_3 m_child1;
    DeepData_3 m_child2;
    DeepData_3 m_child3;
    DeepData_3 m_child4;    
};

struct DeepData_1
{
    DeepData_2 *m_child1;
    
    DeepData_1() :
    m_child1(new DeepData_2())
    {}
};

/*
 type summary add -f "${var._M_dataplus._M_p}" std::string
 type summary add -f "Level 1" "DeepData_1"
 type summary add -f "Level 2" "DeepData_2" -e
 type summary add -f "Level 3" "DeepData_3"
 type summary add -f "Level 4" "DeepData_4"
 type summary add -f "Level 5" "DeepData_5"
 */

int main()
{
    DeepData_1 data1;
    DeepData_2 data2;
    
    return 0; // Set break point at this line.
}