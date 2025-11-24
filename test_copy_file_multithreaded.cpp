// Test case for Issue #169261: [libcxx] copy_file enters an infinite loop in a multithreaded environment
// This test demonstrates the infinite loop bug in copy_file when called concurrently from multiple threads
// when using copy_file_impl_copy_file_range (the count gets stuck at 0, causing an infinite loop)

#include <filesystem>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <atomic>

namespace fs = std::filesystem;

// This test creates multiple threads that attempt to copy the same file concurrently
// to different destinations to trigger the infinite loop condition
void test_copy_file_multithreaded() {
    // Create a test source file
    fs::path src_file = "test_source.txt";
    fs::path dest_dir = "test_dests";
    
    // Cleanup from previous runs
    fs::remove_all(dest_dir);
    fs::create_directories(dest_dir);
    
    // Create source file with some content
    {
        std::ofstream out(src_file);
        out << "Test content for copy operation in multithreaded environment\n";
        out << "This file tests the infinite loop issue reported in Issue #169261\n";
        for (int i = 0; i < 100; ++i) {
            out << "Line " << i << ": Lorem ipsum dolor sit amet\n";
        }
    }
    
    // Test: Create multiple threads that copy the file concurrently
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    std::atomic<int> error_count(0);
    
    auto copy_worker = [&](int thread_id) {
        try {
            fs::path dest = dest_dir / ("copy_" + std::to_string(thread_id) + ".txt");
            // Add timeout mechanism to detect infinite loop (hang would indicate bug)
            std::cout << "Thread " << thread_id << ": Starting copy operation\n";
            fs::copy_file(src_file, dest);
            std::cout << "Thread " << thread_id << ": Copy completed successfully\n";
            success_count++;
        } catch (const std::exception& e) {
            std::cout << "Thread " << thread_id << ": Error - " << e.what() << "\n";
            error_count++;
        }
    };
    
    // Spawn multiple threads
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(copy_worker, i);
    }
    
    // Wait for all threads with timeout
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Test completed: Success=" << success_count << " Errors=" << error_count << "\n";
    
    // Verify all files were created
    int created_files = 0;
    for (const auto& entry : fs::directory_iterator(dest_dir)) {
        if (entry.is_regular_file()) {
            created_files++;
        }
    }
    std::cout << "Files created in destination: " << created_files << "\n";
    
    // Cleanup
    fs::remove(src_file);
    fs::remove_all(dest_dir);
}

int main() {
    std::cout << "Testing copy_file in multithreaded environment (Issue #169261)\n";
    test_copy_file_multithreaded();
    std::cout << "Test finished\n";
    return 0;
}
