#include <vector>

int main()
{

    const char* string1 = "hello world";

    std::vector<int> numbers;
    numbers.push_back(1);  
    numbers.push_back(12);
    numbers.push_back(123);
    numbers.push_back(1234);
    numbers.push_back(12345);
    numbers.push_back(123456);
    numbers.push_back(1234567); // Set break point at this line.
        
    return 0;
}
