#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <type_traits>

// Complex template class to generate more symbols
template<typename T, size_t N>
class ComplexContainer {
private:
    std::vector<T> data;
    std::map<std::string, T> lookup;
    std::mutex mtx;
    
public:
    ComplexContainer() : data(N) {}
    
    void addItem(const std::string& key, const T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        data.push_back(value);
        lookup[key] = value;
    }
    
    T getItem(const std::string& key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = lookup.find(key);
        return (it != lookup.end()) ? it->second : T{};
    }
    
    void processItems() {
        std::lock_guard<std::mutex> lock(mtx);
        std::sort(data.begin(), data.end());
        // Set breakpoint here
        if constexpr (std::is_arithmetic_v<T>) {
            for (auto& item : data) {
                item = item * 2;
            }
        } else {
            // For non-arithmetic types like strings, just reverse
            std::reverse(data.begin(), data.end());
        }
    }
    
    size_t size() const { return data.size(); }
};

// Multiple template instantiations to create more symbols
using IntContainer = ComplexContainer<int, 100>;
using DoubleContainer = ComplexContainer<double, 200>;
using StringContainer = ComplexContainer<std::string, 50>;

class WorkerThread {
private:
    std::unique_ptr<IntContainer> container;
    std::thread worker;
    bool running;
    
public:
    WorkerThread() : container(std::make_unique<IntContainer>()), running(true) {
        worker = std::thread(&WorkerThread::work, this);
    }
    
    ~WorkerThread() {
        running = false;
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    void work() {
        while (running) {
            container->addItem("item_" + std::to_string(rand()), rand() % 1000);
            container->processItems();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    size_t getSize() const { return container->size(); }
};

int main() {
    std::cout << "Complex program starting..." << std::endl;
    
    // Create multiple containers with different types
    IntContainer intContainer;
    DoubleContainer doubleContainer;
    StringContainer stringContainer;
    
    // Add some data
    for (int i = 0; i < 50; ++i) {
        intContainer.addItem("int_" + std::to_string(i), i * 10);
        doubleContainer.addItem("double_" + std::to_string(i), i * 3.14);
        stringContainer.addItem("string_" + std::to_string(i), "value_" + std::to_string(i));
    }
    
    // Process data
    intContainer.processItems();
    doubleContainer.processItems();
    stringContainer.processItems();
    
    // Create worker threads
    std::vector<std::unique_ptr<WorkerThread>> workers;
    for (int i = 0; i < 3; ++i) {
        workers.push_back(std::make_unique<WorkerThread>());
    }
    
    // Let workers run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Print results
    std::cout << "Int container size: " << intContainer.size() << std::endl;
    std::cout << "Double container size: " << doubleContainer.size() << std::endl;
    std::cout << "String container size: " << stringContainer.size() << std::endl;
    
    for (size_t i = 0; i < workers.size(); ++i) {
        std::cout << "Worker " << i << " size: " << workers[i]->getSize() << std::endl;
    }
    
    std::cout << "Complex program finished." << std::endl;
    return 0;
}
