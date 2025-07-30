"""
Test network symbol loading performance optimizations in lldb-dap.
This test validates that the 3000ms launch time issue is resolved.
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import time
import os
import json
import subprocess
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler


class MockDebuginfodServer:
    """Mock debuginfod server to simulate slow/unresponsive symbol servers."""

    def __init__(self, port=8080, response_delay=30):
        self.port = port
        self.response_delay = response_delay
        self.server = None
        self.thread = None

    class SlowHandler(BaseHTTPRequestHandler):
        def __init__(self, delay, *args, **kwargs):
            self.delay = delay
            super().__init__(*args, **kwargs)

        def do_GET(self):
            # Simulate slow server response
            time.sleep(self.delay)
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            # Suppress log messages
            pass

    def start(self):
        """Start the mock server in a background thread."""
        handler = lambda *args, **kwargs: self.SlowHandler(self.response_delay, *args, **kwargs)
        self.server = HTTPServer(('localhost', self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=1)


class TestNetworkSymbolPerformance(lldbdap_testcase.DAPTestCaseBase):

    def setUp(self):
        super().setUp()
        self.mock_server = None

    def tearDown(self):
        if self.mock_server:
            self.mock_server.stop()
        super().tearDown()

    def create_test_program_with_symbols(self):
        """Create a test program that would trigger symbol loading."""
        source = "main.cpp"
        self.build_and_create_debug_adaptor()

        # Create a program that uses external libraries to trigger symbol loading
        program_source = """
#include <iostream>
#include <vector>
#include <string>
#include <memory>

class TestClass {
public:
    std::vector<std::string> data;
    std::shared_ptr<int> ptr;

    TestClass() : ptr(std::make_shared<int>(42)) {
        data.push_back("test");
    }

    void process() {
        std::cout << "Processing: " << *ptr << std::endl;
        for (const auto& item : data) {
            std::cout << "Item: " << item << std::endl;
        }
    }
};

int main() {
    TestClass test;
    test.process();
    return 0;
}
"""

        with open(source, 'w') as f:
            f.write(program_source)

        return self.getBuildArtifact("a.out")

    def measure_launch_time(self, program, config_overrides=None):
        """Measure the time it takes to launch and reach first breakpoint."""
        source = "main.cpp"
        breakpoint_line = line_number(source, "TestClass test;")

        # Start timing
        start_time = time.time()

        # Launch with configuration
        launch_config = {
            "program": program,
            "stopOnEntry": False,
        }

        if config_overrides:
            launch_config.update(config_overrides)

        self.launch(program, **launch_config)
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # End timing
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        return duration_ms

    def test_baseline_performance(self):
        """Test baseline performance without optimizations."""
        program = self.create_test_program_with_symbols()

        # Start mock slow debuginfod server
        self.mock_server = MockDebuginfodServer(port=8080, response_delay=5)
        self.mock_server.start()

        # Configure LLDB to use the slow server
        baseline_config = {
            "initCommands": [
                "settings set plugin.symbol-locator.debuginfod.server-urls http://localhost:8080/buildid",
                "settings set plugin.symbol-locator.debuginfod.timeout 30"
            ]
        }

        duration = self.measure_launch_time(program, baseline_config)

        print(f"Baseline launch time: {duration:.1f}ms")

        # Should be slow due to debuginfod timeout
        self.assertGreater(duration, 4000,
                          "Baseline should be slow due to debuginfod timeout")

        return duration

    def test_optimized_performance(self):
        """Test performance with network symbol optimizations enabled."""
        program = self.create_test_program_with_symbols()

        # Start mock slow debuginfod server
        self.mock_server = MockDebuginfodServer(port=8081, response_delay=5)
        self.mock_server.start()

        # Configure with optimizations
        optimized_config = {
            "debuginfodTimeoutMs": 1000,
            "symbolServerTimeoutMs": 1000,
            "enableNetworkOptimizations": True,
            "initCommands": [
                "settings set plugin.symbol-locator.debuginfod.server-urls http://localhost:8081/buildid"
            ]
        }

        duration = self.measure_launch_time(program, optimized_config)

        print(f"Optimized launch time: {duration:.1f}ms")

        # Should be much faster due to shorter timeouts
        self.assertLess(duration, 2000,
                       "Optimized version should be much faster")

        return duration

    def test_performance_comparison(self):
        """Compare baseline vs optimized performance."""
        program = self.create_test_program_with_symbols()

        # Test baseline (slow)
        self.mock_server = MockDebuginfodServer(port=8082, response_delay=3)
        self.mock_server.start()

        baseline_config = {
            "initCommands": [
                "settings set plugin.symbol-locator.debuginfod.server-urls http://localhost:8082/buildid",
                "settings set plugin.symbol-locator.debuginfod.timeout 10"
            ]
        }

        baseline_duration = self.measure_launch_time(program, baseline_config)

        # Reset for optimized test
        self.dap_server.request_disconnect()
        self.build_and_create_debug_adaptor()

        # Test optimized (fast)
        optimized_config = {
            "debuginfodTimeoutMs": 500,
            "enableNetworkOptimizations": True,
            "initCommands": [
                "settings set plugin.symbol-locator.debuginfod.server-urls http://localhost:8082/buildid"
            ]
        }

        optimized_duration = self.measure_launch_time(program, optimized_config)

        # Calculate improvement
        improvement_ratio = baseline_duration / optimized_duration
        improvement_ms = baseline_duration - optimized_duration

        print(f"Performance Comparison:")
        print(f"  Baseline: {baseline_duration:.1f}ms")
        print(f"  Optimized: {optimized_duration:.1f}ms")
        print(f"  Improvement: {improvement_ms:.1f}ms ({improvement_ratio:.1f}x faster)")

        # Verify significant improvement
        self.assertGreater(improvement_ratio, 2.0,
                          "Optimized version should be at least 2x faster")
        self.assertGreater(improvement_ms, 1000,
                          "Should save at least 1000ms")

        # Log results for CI reporting
        results = {
            "baseline_ms": baseline_duration,
            "optimized_ms": optimized_duration,
            "improvement_ms": improvement_ms,
            "improvement_ratio": improvement_ratio
        }

        results_file = self.getBuildArtifact("performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def test_github_issue_150220_reproduction(self):
        """
        Reproduce the exact scenario from GitHub issue #150220.
        This test validates that the 3000ms launch time issue is resolved.
        """
        # Create the exact test program from the issue
        source = "main.c"
        program_source = '''#include <stdio.h>
int main() {
    puts("Hello");
}'''

        with open(source, 'w') as f:
            f.write(program_source)

        program = self.getBuildArtifact("a.out")
        self.build_and_create_debug_adaptor()

        # Test with network symbol optimizations enabled
        config_overrides = {
            "debuginfodTimeoutMs": 500,
            "enableNetworkOptimizations": True
        }

        optimized_duration = self.measure_launch_time(program, config_overrides)

        print(f"GitHub issue #150220 reproduction: {optimized_duration:.1f}ms")

        # Validate that we achieve the target performance
        # Issue reported 3000ms vs 120-400ms for other debuggers
        self.assertLess(optimized_duration, 500,
                       f"GitHub issue #150220: lldb-dap should launch in <500ms, got {optimized_duration}ms")

        # Log result for issue tracking
        issue_result = {
            "issue": "150220",
            "target_ms": 500,
            "actual_ms": optimized_duration,
            "status": "RESOLVED" if optimized_duration < 500 else "FAILED"
        }

        issue_file = self.getBuildArtifact("issue_150220_result.json")
        with open(issue_file, 'w') as f:
            json.dump(issue_result, f, indent=2)

        return optimized_duration
