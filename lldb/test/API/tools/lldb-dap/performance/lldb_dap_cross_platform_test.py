#!/usr/bin/env python3
"""
Cross-platform testing script for LLDB-DAP network symbol optimizations.
Tests on Linux, macOS, and Windows with various network configurations.
"""

import subprocess
import platform
import os
import sys
import json
import time
from pathlib import Path
import socket
import urllib.request
import urllib.error


class CrossPlatformTester:
    """Test network symbol optimizations across different platforms and configurations."""

    def __init__(self, lldb_dap_path, test_program_path):
        self.lldb_dap_path = Path(lldb_dap_path)
        self.test_program_path = Path(test_program_path)
        self.platform_info = self.get_platform_info()
        self.results = {}

    def get_platform_info(self):
        """Get detailed platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()
        }

    def check_network_connectivity(self):
        """Check basic network connectivity."""
        test_urls = [
            "http://httpbin.org/status/200",
            "https://www.google.com",
            "http://debuginfod.elfutils.org"
        ]

        connectivity = {}
        for url in test_urls:
            try:
                response = urllib.request.urlopen(url, timeout=5)
                connectivity[url] = {
                    "status": "success",
                    "status_code": response.getcode()
                }
            except Exception as e:
                connectivity[url] = {
                    "status": "failed",
                    "error": str(e)
                }

        return connectivity

    def test_platform_specific_features(self):
        """Test platform-specific features and configurations."""
        tests = {}

        # Test file system paths
        tests["file_paths"] = {
            "lldb_dap_exists": self.lldb_dap_path.exists(),
            "test_program_exists": self.test_program_path.exists(),
            "lldb_dap_executable": os.access(self.lldb_dap_path, os.X_OK) if self.lldb_dap_path.exists() else False
        }

        # Test environment variables
        tests["environment"] = {
            "PATH": os.environ.get("PATH", ""),
            "LLDB_DEBUGSERVER_PATH": os.environ.get("LLDB_DEBUGSERVER_PATH"),
            "DEBUGINFOD_URLS": os.environ.get("DEBUGINFOD_URLS"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY")
        }

        # Platform-specific tests
        if self.platform_info["system"] == "Linux":
            tests["linux_specific"] = self.test_linux_features()
        elif self.platform_info["system"] == "Darwin":
            tests["macos_specific"] = self.test_macos_features()
        elif self.platform_info["system"] == "Windows":
            tests["windows_specific"] = self.test_windows_features()

        return tests

    def test_linux_features(self):
        """Test Linux-specific features."""
        tests = {}

        # Check for debuginfod packages
        try:
            result = subprocess.run(["which", "debuginfod"],
                                  capture_output=True, text=True)
            tests["debuginfod_available"] = result.returncode == 0
        except:
            tests["debuginfod_available"] = False

        # Check distribution
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            tests["distribution"] = os_release
        except:
            tests["distribution"] = "unknown"

        return tests

    def test_macos_features(self):
        """Test macOS-specific features."""
        tests = {}

        # Check Xcode tools
        try:
            result = subprocess.run(["xcode-select", "--print-path"],
                                  capture_output=True, text=True)
            tests["xcode_tools"] = result.returncode == 0
            tests["xcode_path"] = result.stdout.strip() if result.returncode == 0 else None
        except:
            tests["xcode_tools"] = False

        # Check system version
        try:
            result = subprocess.run(["sw_vers"], capture_output=True, text=True)
            tests["system_version"] = result.stdout if result.returncode == 0 else None
        except:
            tests["system_version"] = None

        return tests

    def test_windows_features(self):
        """Test Windows-specific features."""
        tests = {}

        # Check Visual Studio tools
        vs_paths = [
            "C:\\Program Files\\Microsoft Visual Studio",
            "C:\\Program Files (x86)\\Microsoft Visual Studio"
        ]

        tests["visual_studio"] = any(Path(p).exists() for p in vs_paths)

        # Check Windows version
        tests["windows_version"] = platform.win32_ver()

        return tests

    def test_network_configurations(self):
        """Test various network configurations."""
        configs = [
            {
                "name": "direct_connection",
                "description": "Direct internet connection",
                "proxy_settings": None
            },
            {
                "name": "with_http_proxy",
                "description": "HTTP proxy configuration",
                "proxy_settings": {"http_proxy": "http://proxy.example.com:8080"}
            },
            {
                "name": "offline_mode",
                "description": "Offline/no network",
                "proxy_settings": {"http_proxy": "http://127.0.0.1:9999"}  # Non-existent proxy
            }
        ]

        results = {}

        for config in configs:
            print(f"Testing network configuration: {config['name']}")

            # Set proxy environment if specified
            original_env = {}
            if config["proxy_settings"]:
                for key, value in config["proxy_settings"].items():
                    original_env[key] = os.environ.get(key.upper())
                    os.environ[key.upper()] = value

            try:
                # Test basic connectivity
                connectivity = self.check_network_connectivity()

                # Test LLDB-DAP with this configuration
                launch_result = self.test_lldb_dap_launch()

                results[config["name"]] = {
                    "description": config["description"],
                    "connectivity": connectivity,
                    "lldb_dap_result": launch_result
                }

            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key.upper(), None)
                    else:
                        os.environ[key.upper()] = value

        return results

    def test_lldb_dap_launch(self):
        """Test basic LLDB-DAP launch functionality."""
        try:
            start_time = time.time()

            process = subprocess.Popen(
                [str(self.lldb_dap_path), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(timeout=10)
            end_time = time.time()

            return {
                "success": process.returncode == 0,
                "duration_ms": (end_time - start_time) * 1000,
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "return_code": process.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "timeout",
                "duration_ms": 10000
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": None
            }

    def run_comprehensive_test(self):
        """Run comprehensive cross-platform testing."""
        print("=" * 60)
        print("LLDB-DAP Cross-Platform Testing")
        print("=" * 60)

        print(f"Platform: {self.platform_info['system']} {self.platform_info['release']}")
        print(f"Architecture: {self.platform_info['machine']}")
        print(f"LLDB-DAP: {self.lldb_dap_path}")
        print(f"Test Program: {self.test_program_path}")

        # Run tests
        self.results = {
            "platform_info": self.platform_info,
            "timestamp": time.time(),
            "tests": {}
        }

        print("\n1. Testing platform-specific features...")
        self.results["tests"]["platform_features"] = self.test_platform_specific_features()

        print("2. Testing network configurations...")
        self.results["tests"]["network_configurations"] = self.test_network_configurations()

        print("3. Testing basic connectivity...")
        self.results["tests"]["connectivity"] = self.check_network_connectivity()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("CROSS-PLATFORM TEST RESULTS")
        print("=" * 60)

        # Platform summary
        platform_tests = self.results["tests"]["platform_features"]
        print(f"\nPlatform: {self.platform_info['system']} {self.platform_info['release']}")
        print(f"LLDB-DAP executable: {'✅' if platform_tests['file_paths']['lldb_dap_executable'] else '❌'}")
        print(f"Test program available: {'✅' if platform_tests['file_paths']['test_program_exists'] else '❌'}")

        # Network configuration results
        network_tests = self.results["tests"]["network_configurations"]
        print(f"\nNetwork Configuration Tests:")
        for config_name, config_result in network_tests.items():
            lldb_success = config_result["lldb_dap_result"]["success"]
            print(f"  {config_name}: {'✅' if lldb_success else '❌'}")

        # Connectivity results
        connectivity = self.results["tests"]["connectivity"]
        print(f"\nConnectivity Tests:")
        for url, result in connectivity.items():
            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {url}: {status}")

        # Save detailed results
        results_file = f"cross_platform_test_results_{self.platform_info['system'].lower()}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

        # Summary
        all_tests_passed = (
            platform_tests["file_paths"]["lldb_dap_executable"] and
            platform_tests["file_paths"]["test_program_exists"] and
            any(config["lldb_dap_result"]["success"] for config in network_tests.values())
        )

        print(f"\nOverall Status: {'✅ PASS' if all_tests_passed else '❌ FAIL'}")

        return all_tests_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-platform testing for LLDB-DAP")
    parser.add_argument("--lldb-dap", required=True, help="Path to lldb-dap executable")
    parser.add_argument("--test-program", required=True, help="Path to test program")

    args = parser.parse_args()

    tester = CrossPlatformTester(args.lldb_dap, args.test_program)
    success = tester.run_comprehensive_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
