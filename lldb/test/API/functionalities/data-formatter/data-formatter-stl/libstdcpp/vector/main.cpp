#include <string>
#include <vector>
typedef std::vector<int> int_vect;
typedef std::vector<std::string> string_vect;

int main()
{
    int_vect numbers;
    numbers.push_back(1);  // Set break point at this line.
    numbers.push_back(12);  // Set break point at this line.
    numbers.push_back(123);
    numbers.push_back(1234);
    numbers.push_back(12345);  // Set break point at this line.
    numbers.push_back(123456);
    numbers.push_back(1234567);
    
    numbers.clear();  // Set break point at this line.
    
    numbers.push_back(7);  // Set break point at this line.

    string_vect strings;  // Set break point at this line.
    strings.push_back(std::string("goofy"));
    strings.push_back(std::string("is"));
    strings.push_back(std::string("smart"));
    
    strings.push_back(std::string("!!!"));  // Set break point at this line.
    
    strings.clear();  // Set break point at this line.
    
    return 0;// Set break point at this line.
}
