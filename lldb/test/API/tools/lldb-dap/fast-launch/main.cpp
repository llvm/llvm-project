#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "Fast launch test program starting..." << std::endl;

    // Create some variables for debugging
    int counter = 0;
    std::vector<std::string> items = {"apple", "banana", "cherry"};

    for (const auto& item : items) {
        counter++;
        std::cout << "Item " << counter << ": " << item << std::endl; // Set breakpoint here
    }

    std::cout << "Program completed with " << counter << " items processed." << std::endl;
    return 0;
}
