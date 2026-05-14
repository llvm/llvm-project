#!/usr/bin/env python3
"""
Interactive wizard for configuring the LLVM Instrumentor pass.

This script helps users create custom instrumentation configurations by:
1. Generating a default config file using opt
2. Presenting available instrumentation options interactively
3. Allowing users to enable/disable specific instrumentation opportunities
4. Saving the customized configuration to a JSON file
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Any, Optional, Tuple


class InstrumentorConfigWizard:
    def __init__(self, opt_path: str = None):
        """Initialize the wizard with the path to opt."""
        self.opt_path = opt_path or self.find_opt()
        self.config = {}
        self.enabled_opportunities = set()
        self.same_pre_post = True
        self.navigation_stack = []

    def find_opt(self) -> str:
        """Find the opt binary in the build directory."""
        # Try common locations relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))

        # Check build/bin/opt
        opt_candidates = [
            os.path.join(repo_root, "build", "bin", "opt"),
            os.path.join(repo_root, "build", "Debug", "bin", "opt"),
            os.path.join(repo_root, "build", "Release", "bin", "opt"),
            "opt",  # Try system PATH
        ]

        for candidate in opt_candidates:
            if os.path.exists(candidate):
                return candidate
            # Check if it's in PATH
            try:
                subprocess.run(
                    [candidate, "--version"], capture_output=True, check=True, timeout=5
                )
                return candidate
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                continue

        raise FileNotFoundError(
            "Could not find 'opt' binary. Please specify the path using --opt-path"
        )

    def generate_default_config(self) -> Dict[str, Any]:
        """Generate a default configuration by running opt."""
        print(f"Generating default configuration using: {self.opt_path}")

        # Create a minimal LLVM IR module to trigger config generation
        minimal_ir = """
define i32 @main() {
  %1 = alloca i32
  store i32 0, ptr %1
  %2 = load i32, ptr %1
  ret i32 %2
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            ir_file = os.path.join(tmpdir, "input.ll")
            config_file = os.path.join(tmpdir, "config.json")

            # Write minimal IR
            with open(ir_file, "w") as f:
                f.write(minimal_ir)

            # Run opt with instrumentor to generate config
            try:
                cmd = [
                    self.opt_path,
                    "-passes=instrumentor",
                    f"-instrumentor-write-config-file={config_file}",
                    "-disable-output",
                    ir_file,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    print(
                        f"Warning: opt returned non-zero exit code: {result.returncode}"
                    )
                    if result.stderr:
                        print(f"stderr: {result.stderr}")

                # Read the generated config
                if not os.path.exists(config_file):
                    raise FileNotFoundError(
                        f"Config file was not generated at {config_file}"
                    )

                with open(config_file, "r") as f:
                    config = json.load(f)

                print("✓ Default configuration generated successfully\n")
                return config

            except subprocess.TimeoutExpired:
                raise RuntimeError("opt command timed out")
            except Exception as e:
                raise RuntimeError(f"Failed to generate config: {e}")

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system("clear" if os.name != "nt" else "cls")

    def print_section_header(self, title: str):
        """Print a formatted section header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_option(self, index: int, name: str, description: str, enabled: bool):
        """Print a formatted option."""
        status = "[X]" if enabled else "[ ]"
        print(f"  {index:2d}. {status} {name:30s} - {description}")

    def get_user_choice(
        self, prompt: str, valid_choices: List[str] = None, allow_back: bool = True
    ) -> Optional[str]:
        """Get user input with validation."""
        while True:
            try:
                nav_hint = " (b=back, q=quit)" if allow_back else " (q=quit)"
                choice = input(prompt + nav_hint + ": ").strip().lower()

                if choice == "q":
                    confirm = input("Really quit? (y/n): ").strip().lower()
                    if confirm == "y":
                        print("\nWizard cancelled by user.")
                        sys.exit(0)
                    continue

                if choice == "b" and allow_back:
                    return "BACK"

                if not choice:
                    return ""

                if valid_choices:
                    if choice in valid_choices:
                        return choice
                    print(f"Please enter one of: {', '.join(valid_choices)}")
                else:
                    return choice

            except KeyboardInterrupt:
                print("\n\nWizard interrupted by user.")
                sys.exit(0)

    def get_all_opportunity_types(self) -> List[Tuple[str, str]]:
        """Extract all unique opportunity types from the config."""
        opportunities = []
        seen = set()

        for location in [
            "function_pre",
            "function_post",
            "instruction_pre",
            "instruction_post",
        ]:
            if location not in self.config:
                continue

            for opp_name in self.config[location].keys():
                if opp_name not in seen:
                    seen.add(opp_name)
                    # Get description from first occurrence
                    opp_config = self.config[location][opp_name]
                    desc = "No description available"

                    # Try to find a description from any field
                    for key, value in opp_config.items():
                        if key == "enabled":
                            continue
                        if key.endswith(".description") and value:
                            desc = value
                            break

                    opportunities.append((opp_name, desc))

        return sorted(opportunities)

    def select_opportunities(self) -> bool:
        """Let user select which instrumentation opportunities to enable."""
        while True:
            self.clear_screen()
            self.print_section_header("Step 1: Select Instrumentation Types")

            opportunities = self.get_all_opportunity_types()

            print("\nSelect which types of instrumentation you want to configure:")
            print(
                "(You can toggle individual arguments for each type in the next steps)\n"
            )

            for idx, (opp_name, opp_desc) in enumerate(opportunities, 1):
                enabled = opp_name in self.enabled_opportunities
                self.print_option(idx, opp_name, opp_desc, enabled)

            print("\nCommands:")
            print("  - Enter numbers (space-separated) to toggle opportunities")
            print("  - 'all' to enable all, 'none' to disable all")
            print("  - Press Enter when done to continue")

            choice = self.get_user_choice("\nYour choice", allow_back=False)

            if choice == "BACK":
                continue
            elif choice == "":
                if not self.enabled_opportunities:
                    print("\n⚠ Please enable at least one instrumentation type!")
                    input("Press Enter to continue...")
                    continue
                return True
            elif choice == "all":
                self.enabled_opportunities = {opp[0] for opp in opportunities}
            elif choice == "none":
                self.enabled_opportunities.clear()
            else:
                try:
                    indices = [int(x) for x in choice.split()]
                    for idx in indices:
                        if 1 <= idx <= len(opportunities):
                            opp_name = opportunities[idx - 1][0]
                            if opp_name in self.enabled_opportunities:
                                self.enabled_opportunities.remove(opp_name)
                            else:
                                self.enabled_opportunities.add(opp_name)
                except ValueError:
                    print(
                        "\n⚠ Invalid input. Please enter numbers separated by spaces."
                    )
                    input("Press Enter to continue...")

    def configure_pre_post_mode(self) -> bool:
        """Ask if PRE and POST should have the same configuration."""
        while True:
            self.clear_screen()
            self.print_section_header("Step 2: PRE vs POST Configuration")

            print("\nInstrumentation can happen at two points:")
            print("  - PRE:  Before the instrumented operation")
            print("  - POST: After the instrumented operation")
            print("\nFor example, for a load instruction:")
            print("  - PRE:  Can inspect/modify the pointer before the load")
            print("  - POST: Can inspect/modify the loaded value after the load")

            print(
                f"\nCurrent mode: {'SAME configuration for PRE and POST' if self.same_pre_post else 'DIFFERENT configurations'}"
            )

            choice = self.get_user_choice(
                "\nUse same configuration for PRE and POST? (y/n/Enter to keep)",
                valid_choices=["y", "yes", "n", "no", ""],
            )

            if choice == "BACK":
                return False
            elif choice in ["y", "yes"]:
                self.same_pre_post = True
                return True
            elif choice in ["n", "no"]:
                self.same_pre_post = False
                return True
            elif choice == "":
                return True

    def configure_base_options(self) -> bool:
        """Configure base/global options."""
        while True:
            self.clear_screen()
            self.print_section_header("Step 3: Base Configuration")

            if "configuration" not in self.config:
                self.config["configuration"] = {}

            base_config = self.config["configuration"]

            # Display current settings
            print("\nCurrent settings:")
            print(
                f"  1. Runtime prefix:         {base_config.get('runtime_prefix', '__instrumentor_')}"
            )
            print(
                f"  2. Demangle function names: {base_config.get('demangle_function_names', True)}"
            )
            print(
                f"  3. Target regex:           {base_config.get('target_regex', '(none)')}"
            )
            print(
                f"  4. Host (CPU) enabled:     {base_config.get('host_enabled', True)}"
            )
            print(
                f"  5. GPU enabled:            {base_config.get('gpu_enabled', True)}"
            )

            print("\nEnter option number to modify, or press Enter to continue")
            choice = self.get_user_choice("Option")

            if choice == "BACK":
                return False
            elif choice == "":
                return True
            elif choice == "1":
                new_prefix = input("Enter runtime prefix: ").strip()
                if new_prefix:
                    base_config["runtime_prefix"] = new_prefix
            elif choice == "2":
                demangle = self.get_user_choice(
                    "Demangle function names? (y/n)", ["y", "n"], allow_back=False
                )
                if demangle:
                    base_config["demangle_function_names"] = demangle == "y"
            elif choice == "3":
                new_regex = input("Enter target regex (empty for none): ").strip()
                base_config["target_regex"] = new_regex
            elif choice == "4":
                host = self.get_user_choice(
                    "Enable host instrumentation? (y/n)", ["y", "n"], allow_back=False
                )
                if host:
                    base_config["host_enabled"] = host == "y"
            elif choice == "5":
                gpu = self.get_user_choice(
                    "Enable GPU instrumentation? (y/n)", ["y", "n"], allow_back=False
                )
                if gpu:
                    base_config["gpu_enabled"] = gpu == "y"

    def configure_opportunity_args(
        self, opp_name: str, location: str, step_prefix: str = "Step 4"
    ) -> bool:
        """Configure arguments for a specific opportunity at a location."""
        while True:
            self.clear_screen()
            location_desc = "PRE (before)" if "pre" in location else "POST (after)"
            self.print_section_header(
                f"{step_prefix}: Configure {opp_name} - {location_desc}"
            )

            if location not in self.config or opp_name not in self.config[location]:
                print(f"\n⚠ {opp_name} not found in {location}")
                input("Press Enter to continue...")
                return True

            opp_config = self.config[location][opp_name]

            # Show enable/disable status
            enabled = opp_config.get("enabled", False)
            print(f"\nInstrumentation: {'ENABLED ✓' if enabled else 'DISABLED ✗'}")

            # Collect arguments
            args = []
            for key, value in sorted(opp_config.items()):
                if (
                    key == "enabled"
                    or key.endswith(".description")
                    or key.endswith(".replace")
                ):
                    continue
                desc = opp_config.get(f"{key}.description", "No description")
                can_replace = f"{key}.replace" in opp_config
                replace_enabled = (
                    opp_config.get(f"{key}.replace", False) if can_replace else False
                )
                args.append((key, value, desc, can_replace, replace_enabled))

            if args:
                print("\nAvailable arguments:")
                for idx, (
                    arg_name,
                    arg_enabled,
                    arg_desc,
                    can_replace,
                    replace_enabled,
                ) in enumerate(args, 1):
                    status = "[X]" if arg_enabled else "[ ]"
                    if can_replace:
                        replace_status = "REPLACE" if replace_enabled else "observe"
                        replace_mark = f" [replaceable: {replace_status}]"
                    else:
                        replace_mark = ""
                    print(
                        f"  {idx:2d}. {status} {arg_name:25s} - {arg_desc}{replace_mark}"
                    )

            print("\nCommands:")
            print("  - 'e' to toggle enabled/disabled")
            print("  - Enter numbers (space-separated) to toggle arguments")
            print(
                "  - 'r <num>' to toggle replacement for replaceable argument (e.g., 'r 1')"
            )
            print("  - 'all' to enable all args, 'none' to disable all args")
            print("  - Press Enter when done")

            choice = self.get_user_choice("\nYour choice")

            if choice == "BACK":
                return False
            elif choice == "":
                return True
            elif choice == "e":
                opp_config["enabled"] = not opp_config["enabled"]
            elif choice == "all":
                for arg_name, _, _, _, _ in args:
                    opp_config[arg_name] = True
            elif choice == "none":
                for arg_name, _, _, _, _ in args:
                    opp_config[arg_name] = False
            elif choice.startswith("r "):
                # Toggle replacement
                try:
                    parts = choice.split()
                    if len(parts) == 2:
                        idx = int(parts[1])
                        if 1 <= idx <= len(args):
                            arg_name, _, _, can_replace, _ = args[idx - 1]
                            if can_replace:
                                replace_key = f"{arg_name}.replace"
                                opp_config[replace_key] = not opp_config.get(
                                    replace_key, False
                                )
                            else:
                                print(f"\n⚠ Argument '{arg_name}' is not replaceable.")
                                input("Press Enter to continue...")
                        else:
                            print(f"\n⚠ Invalid argument number: {idx}")
                            input("Press Enter to continue...")
                    else:
                        print("\n⚠ Usage: r <number>")
                        input("Press Enter to continue...")
                except ValueError:
                    print("\n⚠ Invalid input for replacement toggle.")
                    input("Press Enter to continue...")
            else:
                try:
                    indices = [int(x) for x in choice.split()]
                    for idx in indices:
                        if 1 <= idx <= len(args):
                            arg_name = args[idx - 1][0]
                            opp_config[arg_name] = not opp_config[arg_name]
                except ValueError:
                    print("\n⚠ Invalid input.")
                    input("Press Enter to continue...")

    def configure_locations(self) -> bool:
        """Configure all enabled opportunities for PRE and optionally POST."""
        # First, disable all opportunities that are not in enabled_opportunities
        for location in [
            "function_pre",
            "function_post",
            "instruction_pre",
            "instruction_post",
        ]:
            if location not in self.config:
                continue
            for opp_name, opp_config in self.config[location].items():
                if opp_name not in self.enabled_opportunities:
                    opp_config["enabled"] = False

        # Configure PRE locations
        step_num = 4
        for idx, opp_name in enumerate(sorted(self.enabled_opportunities), 1):
            # Try function_pre first, then instruction_pre
            location = None
            if (
                "function_pre" in self.config
                and opp_name in self.config["function_pre"]
            ):
                location = "function_pre"
            elif (
                "instruction_pre" in self.config
                and opp_name in self.config["instruction_pre"]
            ):
                location = "instruction_pre"

            if location:
                if not self.configure_opportunity_args(opp_name, location):
                    return False

        # If same config, copy PRE to POST
        if self.same_pre_post:
            for opp_name in self.enabled_opportunities:
                # Copy from PRE to POST
                if (
                    "function_pre" in self.config
                    and opp_name in self.config["function_pre"]
                ):
                    if (
                        "function_post" in self.config
                        and opp_name in self.config["function_post"]
                    ):
                        pre_config = self.config["function_pre"][opp_name]
                        post_config = self.config["function_post"][opp_name]
                        # Copy enabled and argument settings
                        post_config["enabled"] = pre_config.get("enabled", False)
                        for key in pre_config:
                            if not key.endswith(".description") and key != "enabled":
                                if key in post_config:
                                    post_config[key] = pre_config[key]

                if (
                    "instruction_pre" in self.config
                    and opp_name in self.config["instruction_pre"]
                ):
                    if (
                        "instruction_post" in self.config
                        and opp_name in self.config["instruction_post"]
                    ):
                        pre_config = self.config["instruction_pre"][opp_name]
                        post_config = self.config["instruction_post"][opp_name]
                        post_config["enabled"] = pre_config.get("enabled", False)
                        for key in pre_config:
                            if not key.endswith(".description") and key != "enabled":
                                if key in post_config:
                                    post_config[key] = pre_config[key]
        else:
            # Configure POST locations separately
            for opp_name in sorted(self.enabled_opportunities):
                location = None
                if (
                    "function_post" in self.config
                    and opp_name in self.config["function_post"]
                ):
                    location = "function_post"
                elif (
                    "instruction_post" in self.config
                    and opp_name in self.config["instruction_post"]
                ):
                    location = "instruction_post"

                if location:
                    if not self.configure_opportunity_args(opp_name, location):
                        return False

        return True

    def generate_runtime_stubs(self, config_path: str, stub_path: str) -> bool:
        """Generate runtime stub file using the configuration."""
        print(f"\nGenerating runtime stubs using: {self.opt_path}")

        # Create a minimal LLVM IR module
        minimal_ir = """
define i32 @main() {
  %1 = alloca i32
  store i32 0, ptr %1
  %2 = load i32, ptr %1
  ret i32 %2
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            ir_file = os.path.join(tmpdir, "input.ll")
            temp_config = os.path.join(tmpdir, "temp_config.json")

            # Write minimal IR
            with open(ir_file, "w") as f:
                f.write(minimal_ir)

            # Create a temporary config with stub file set
            temp_cfg = self.config.copy()
            if "configuration" not in temp_cfg:
                temp_cfg["configuration"] = {}
            temp_cfg["configuration"]["runtime_stubs_file"] = stub_path

            with open(temp_config, "w") as f:
                json.dump(temp_cfg, f, indent=2)

            # Run opt with instrumentor to generate stubs
            try:
                cmd = [
                    self.opt_path,
                    "-passes=instrumentor",
                    f"-instrumentor-read-config-file={temp_config}",
                    "-disable-output",
                    ir_file,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    print(
                        f"Warning: opt returned non-zero exit code: {result.returncode}"
                    )
                    if result.stderr:
                        print(f"stderr: {result.stderr}")

                # Check if stub file was generated
                if os.path.exists(stub_path):
                    print(f"✓ Runtime stubs generated: {stub_path}")
                    return True
                else:
                    print(f"✗ Stub file was not generated")
                    return False

            except subprocess.TimeoutExpired:
                print("✗ opt command timed out")
                return False
            except Exception as e:
                print(f"✗ Failed to generate stubs: {e}")
                return False

    def review_and_save(self, output_path: str) -> bool:
        """Review configuration and save."""
        stub_path = None

        while True:
            self.clear_screen()
            self.print_section_header("Step 5: Review and Save")

            print("\nEnabled instrumentation types:")
            for opp in sorted(self.enabled_opportunities):
                print(f"  ✓ {opp}")

            print(
                f"\nPRE/POST mode: {'Same configuration' if self.same_pre_post else 'Different configurations'}"
            )
            print(
                f"Runtime prefix: {self.config.get('configuration', {}).get('runtime_prefix', '__instrumentor_')}"
            )
            print(f"\nConfiguration file: {output_path}")
            if stub_path:
                print(f"Runtime stubs file: {stub_path}")

            print("\nCommands:")
            print("  - 's' to save configuration and finish")
            print("  - 'g' to generate runtime stub file (optional)")
            print("  - 'p' to specify different output path")
            print("  - 'b' to go back and modify settings")

            choice = self.get_user_choice(
                "\nYour choice", valid_choices=["s", "g", "p", "b", ""]
            )

            if choice == "BACK" or choice == "b":
                return False
            elif choice == "s":
                try:
                    # Remove runtime_stubs_file from config before saving
                    config_to_save = json.loads(json.dumps(self.config))
                    if "configuration" in config_to_save:
                        config_to_save["configuration"].pop("runtime_stubs_file", None)
                        config_to_save["configuration"].pop(
                            "runtime_stubs_file.description", None
                        )

                    with open(output_path, "w") as f:
                        json.dump(config_to_save, f, indent=2)
                    print(f"\n✓ Configuration saved to: {output_path}")

                    # Generate stubs if requested
                    if stub_path:
                        self.generate_runtime_stubs(output_path, stub_path)

                    return True
                except Exception as e:
                    print(f"\n✗ Failed to save configuration: {e}")
                    input("Press Enter to continue...")
            elif choice == "g":
                print("\nGenerate runtime stub file")
                print("This creates a C/C++ file with stub implementations of the")
                print(
                    "instrumentation runtime functions that you can use as a template."
                )

                default_stub = output_path.rsplit(".", 1)[0] + "_stubs.c"
                stub_input = input(
                    f"\nStub file path (default: {default_stub}): "
                ).strip()
                stub_path = stub_input if stub_input else default_stub
                print(f"Will generate stubs to: {stub_path}")
                input("Press Enter to continue...")
            elif choice == "p":
                new_path = input("Enter configuration output path: ").strip()
                if new_path:
                    output_path = new_path
            elif choice == "":
                continue

    def run_interactive(self, output_path: str):
        """Run the interactive configuration wizard."""
        self.clear_screen()
        print("=" * 70)
        print("  LLVM Instrumentor Configuration Wizard")
        print("=" * 70)
        print(
            "\nThis wizard will help you create a custom instrumentation configuration."
        )
        print("You can enable/disable instrumentation opportunities and configure")
        print("what information is passed to the runtime functions.")
        print("\nNavigation: Use 'b' to go back, 'q' to quit at any prompt.")
        input("\nPress Enter to continue...")

        # Generate or load config
        try:
            self.config = self.generate_default_config()
        except Exception as e:
            print(f"Error: {e}")
            return False

        # State machine for navigation
        state = 0
        while True:
            if state == 0:  # Select opportunities
                if self.select_opportunities():
                    state = 1
            elif state == 1:  # PRE/POST mode
                if self.configure_pre_post_mode():
                    state = 2
                else:
                    state = 0
            elif state == 2:  # Base configuration
                if self.configure_base_options():
                    state = 3
                else:
                    state = 1
            elif state == 3:  # Configure locations
                if self.configure_locations():
                    state = 4
                else:
                    state = 2
            elif state == 4:  # Review and save
                if self.review_and_save(output_path):
                    return True
                else:
                    state = 3


def main():
    parser = argparse.ArgumentParser(
        description="Interactive wizard for configuring LLVM Instrumentor pass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  %(prog)s

  # Specify custom output location
  %(prog)s -o my_config.json

  # Use specific opt binary
  %(prog)s --opt-path /path/to/opt

  # Load existing config and modify it
  %(prog)s --input existing_config.json -o modified_config.json
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        default="instrumentor_config.json",
        help="Output configuration file (default: instrumentor_config.json)",
    )

    parser.add_argument(
        "--opt-path", help="Path to the opt binary (default: auto-detect)"
    )

    parser.add_argument(
        "--input", help="Load and modify an existing configuration file"
    )

    args = parser.parse_args()

    try:
        wizard = InstrumentorConfigWizard(opt_path=args.opt_path)

        # Load existing config if provided
        if args.input:
            print(f"Loading existing configuration from: {args.input}")
            with open(args.input, "r") as f:
                wizard.config = json.load(f)
            print("✓ Configuration loaded\n")
            # Extract enabled opportunities from loaded config
            for location in [
                "function_pre",
                "function_post",
                "instruction_pre",
                "instruction_post",
            ]:
                if location in wizard.config:
                    for opp_name, opp_config in wizard.config[location].items():
                        if opp_config.get("enabled", False):
                            wizard.enabled_opportunities.add(opp_name)

        success = wizard.run_interactive(args.output)

        if success:
            print("\n" + "=" * 70)
            print("Configuration complete!")
            print("=" * 70)
            print(f"\nTo use this configuration with opt:")
            print(f"  opt -passes=instrumentor \\")
            print(f"      -instrumentor-read-config-file={args.output} \\")
            print(f"      input.ll -S -o output.ll")
            print(f"\nTo use with clang:")
            print(f"  clang -mllvm -enable-instrumentor \\")
            print(f"        -mllvm -instrumentor-read-config-file={args.output} \\")
            print(f"        input.c -o output")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
